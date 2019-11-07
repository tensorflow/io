/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/core/kernels/io_stream.h"

namespace tensorflow {
namespace data {

// See http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
struct WAVHeader {
  char riff[4];              // RIFF Chunk ID: "RIFF"
  int32 riff_size;           // RIFF chunk size: 4 + n (file size - 8)
  char wave[4];              // WAVE ID: "WAVE"
  char fmt[4];               // fmt Chunk ID: "fmt "
  int32 fmt_size;            // fmt Chunk size: 16, 18, or 40
  int16 wFormatTag;          // Format code: WAVE_FORMAT_PCM (1) for PCM. WAVE_FORMAT_EXTENSIBLE (0xFFFE) for SubFormat
  int16 nChannels;           // Number of channels
  int32 nSamplesPerSec;      // Sampling rate
  int32 nAvgBytesPerSec;     // Data rate
  int16 nBlockAlign;         // Data block size (bytes)
  int16 wBitsPerSample;      // Bits per sample
};

struct ExtensionHeader {
  int16 cbSize;              // Size of the extension (0 or 22)
  int16 wValidBitsPerSample; // Number of valid bits
  int32 dwChannelMask;       // Speaker position mask
  char SubFormat[16];        // GUID
};

struct DataHeader {
  char mark[4];
  int32 size;
};
Status ValidateWAVHeader(struct WAVHeader *header) {
  if (memcmp(header->riff, "RIFF", 4) != 0) {
    return errors::InvalidArgument("WAV file must starts with `RIFF`");
  }
  if (memcmp(header->wave, "WAVE", 4) != 0) {
    return errors::InvalidArgument("WAV file must contains riff type `WAVE`");
  }
  if (memcmp(header->fmt, "fmt ", 4) != 0) {
    return errors::InvalidArgument("WAV file must contains `fmt ` mark");
  }
  if (header->fmt_size != 16 && header->fmt_size != 18 && header->fmt_size != 40) {
    return errors::InvalidArgument("WAV file must have `fmt_size ` 16, 18, or 40, received: ", header->fmt_size);
  }
  if (header->wFormatTag != 1 && header->wFormatTag != static_cast<int16>(0xFFFE)) {
    return errors::InvalidArgument("WAV file must have `wFormatTag` 1 or 0xFFFE, received: ", header->wFormatTag);
  }
  if (header->nChannels <= 0) {
    return errors::InvalidArgument("WAV file have invalide channels: ", header->nChannels);
  }
  return Status::OK();
}


class WAVReadableResource : public ResourceBase {
 public:
  WAVReadableResource(Env* env) : env_(env) {}
  ~WAVReadableResource() {}

  Status Init(const string& input) {
    const string& filename = input;
    file_.reset(new SizedRandomAccessFile(env_, filename, nullptr, 0));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    StringPiece result;
    TF_RETURN_IF_ERROR(file_->Read(0, sizeof(header_), &result, (char *)(&header_)));
    header_length_ = sizeof(header_);
    int64 fmt_position = 12;
    while (memcmp(header_.fmt, "fmt ", 4) != 0) {
      // Skip JUNK/bext/etc field.
      if (memcmp(header_.fmt, "JUNK", 4) != 0 &&
          memcmp(header_.fmt, "bext", 4) != 0 &&
          memcmp(header_.fmt, "iXML", 4) != 0 &&
          memcmp(header_.fmt, "qlty", 4) != 0 &&
          memcmp(header_.fmt, "mext", 4) != 0 &&
          memcmp(header_.fmt, "levl", 4) != 0 &&
          memcmp(header_.fmt, "link", 4) != 0 &&
          memcmp(header_.fmt, "axml", 4) != 0) {
        return errors::InvalidArgument("unexpected field: ", header_.fmt);
      }
      int32 size_of_chunk = 4 + 4 + header_.fmt_size;
      if (header_.fmt_size % 2 == 1) {
        size_of_chunk += 1;
      }
      fmt_position += size_of_chunk;
      // Re-read the header
      TF_RETURN_IF_ERROR(file_->Read(fmt_position, sizeof(header_) - 12, &result, (char *)(&header_) + 12));
      header_length_ = fmt_position + sizeof(header_) - 12;
    }

    TF_RETURN_IF_ERROR(ValidateWAVHeader(&header_));
    if (header_.riff_size + 8 != file_size_) {
      // corrupted file?
    }
    int64 filesize = header_.riff_size + 8;
    int64 position = header_length_ + header_.fmt_size - 16;

    int64 nSamples = 0;
    do {
      struct DataHeader head;
      TF_RETURN_IF_ERROR(file_->Read(position, sizeof(head), &result, (char *)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        // Data should be block aligned
        // bytes = nSamples * nBlockAlign
        if (head.size % header_.nBlockAlign != 0) {
          return errors::InvalidArgument("data chunk should be block aligned (", header_.nBlockAlign, "), received: ", head.size);
        }
        nSamples += head.size / header_.nBlockAlign;
      }
      position += head.size;
    } while (position < filesize);

    switch (header_.wBitsPerSample) {
    case 8:
      dtype_ = DT_INT8;
      break;
    case 16:
      dtype_ = DT_INT16;
      break;
    case 24:
      dtype_ = DT_INT32;
      break;
    default:
      return errors::InvalidArgument("unsupported wBitsPerSample: ", header_.wBitsPerSample);
    }

    shape_ = TensorShape({nSamples, header_.nChannels});

    return Status::OK();
  }
  Status Spec(TensorShape* shape, DataType* dtype, int32 *rate) {
    *shape = shape_;
    *dtype = dtype_;
    *rate = header_.nSamplesPerSec;
    return Status::OK();
  }

  Status Peek(const int64 start, const int64 stop, TensorShape* shape) {
    int64 sample_stop = (stop < 0) ? (shape_.dim_size(0)) : (stop < shape_.dim_size(0) ? stop : shape_.dim_size(0));
    int64 sample_start = (start >= sample_stop) ? sample_stop : start;
    *shape = TensorShape({sample_stop - sample_start, header_.nChannels});
    return Status::OK();
  }
  Status Read(const int64 start, Tensor* value) {
    const int64 sample_start = start;
    const int64 sample_stop = start + value->shape().dim_size(0);

    int64 sample_offset = 0;
    if (header_.riff_size + 8 != file_size_) {
      // corrupted file?
    }
    int64 filesize = header_.riff_size + 8;
    int64 position = header_length_ + header_.fmt_size - 16;
    do {
      StringPiece result;
      struct DataHeader head;
      TF_RETURN_IF_ERROR(file_->Read(position, sizeof(head), &result, (char *)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        // Already checked the alignment
        int64 block_sample_start = sample_offset;
        int64 block_sample_stop = sample_offset + head.size / header_.nBlockAlign;
        // only read if block_sample_start and block_sample_stop within range
        if (sample_start < block_sample_stop && sample_stop > block_sample_start) {
          int64 read_sample_start = (block_sample_start > sample_start ? block_sample_start : sample_start);
          int64 read_sample_stop = (block_sample_stop < sample_stop ? block_sample_stop : sample_stop);
          int64 read_bytes_start = position + (read_sample_start - block_sample_start) * header_.nBlockAlign;
          int64 read_bytes_stop = position + (read_sample_stop - block_sample_start) * header_.nBlockAlign;
          string buffer;
          buffer.resize(read_bytes_stop - read_bytes_start);
          TF_RETURN_IF_ERROR(file_->Read(read_bytes_start, read_bytes_stop - read_bytes_start, &result, &buffer[0]));
          switch (header_.wBitsPerSample) {
          case 8:
            if (header_.wBitsPerSample * header_.nChannels != header_.nBlockAlign * 8) {
              return errors::InvalidArgument("unsupported wBitsPerSample and header.nBlockAlign: ", header_.wBitsPerSample, ", ", header_.nBlockAlign);
            }
            memcpy((char *)(value->flat<int8>().data()) + ((read_sample_start - sample_start) * header_.nBlockAlign), &buffer[0], (read_bytes_stop - read_bytes_start));
            break;
          case 16:
            if (header_.wBitsPerSample * header_.nChannels != header_.nBlockAlign * 8) {
              return errors::InvalidArgument("unsupported wBitsPerSample and header.nBlockAlign: ", header_.wBitsPerSample, ", ", header_.nBlockAlign);
            }
            memcpy((char *)(value->flat<int16>().data()) + ((read_sample_start - sample_start) * header_.nBlockAlign), &buffer[0], (read_bytes_stop - read_bytes_start));
            break;
          case 24:
            // NOTE: The conversion is from signed integer 24 to signed integer 32 (left shift 8 bits)
            if (header_.wBitsPerSample * header_.nChannels != header_.nBlockAlign * 8) {
              return errors::InvalidArgument("unsupported wBitsPerSample and header.nBlockAlign: ", header_.wBitsPerSample, ", ", header_.nBlockAlign);
            }
            for (int64 i = read_sample_start; i < read_sample_stop; i++) {
              for (int64 j = 0; j < header_.nChannels; j++) {
                char *data_p = (char *)(value->flat<int32>().data() + ((i - sample_start) * header_.nChannels + j));
                char *read_p = (char *)(&buffer[((i - read_sample_start) * header_.nBlockAlign)]) + 3 * j;
                data_p[3] = read_p[2];
                data_p[2] = read_p[1];
                data_p[1] = read_p[0];
                data_p[0] = 0x00;
              }
            }
            break;
          default:
            return errors::InvalidArgument("unsupported wBitsPerSample and header.nBlockAlign: ", header_.wBitsPerSample, ", ", header_.nBlockAlign);
          }
        }
        sample_offset = block_sample_stop;
      }
      position += head.size;
    } while (position < filesize);

    return Status::OK();
  }
  string DebugString() const override {
    return "WAVReadableResource";
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  DataType dtype_;
  TensorShape shape_;
  struct WAVHeader header_;
  int64 header_length_;
};

class WAVReadableInitOp : public ResourceOpKernel<WAVReadableResource> {
 public:
  explicit WAVReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<WAVReadableResource>(context) {
    env_ = context->env();
  }
 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<WAVReadableResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    OP_REQUIRES_OK(context, resource_->Init(input_tensor->scalar<string>()()));
  }
  Status CreateResource(WAVReadableResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new WAVReadableResource(env_);
    return Status::OK();
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};


class WAVReadableSpecOp : public OpKernel {
 public:
  explicit WAVReadableSpecOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    WAVReadableResource* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    TensorShape shape;
    DataType dtype;
    int32 rate;
    OP_REQUIRES_OK(context, resource->Spec(&shape, &dtype, &rate));

    Tensor* shape_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({2}), &shape_tensor));
    shape_tensor->flat<int64>()(0) = shape.dim_size(0);
    shape_tensor->flat<int64>()(1) = shape.dim_size(1);

    Tensor* dtype_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &dtype_tensor));
    dtype_tensor->scalar<int64>()() = dtype;

    Tensor* rate_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({}), &rate_tensor));
    rate_tensor->scalar<int32>()() = rate;
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
class WAVReadableReadOp : public OpKernel {
 public:
  explicit WAVReadableReadOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    WAVReadableResource* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    int64 start = start_tensor->scalar<int64>()();

    const Tensor* stop_tensor;
    OP_REQUIRES_OK(context, context->input("stop", &stop_tensor));
    int64 stop = stop_tensor->scalar<int64>()();
    TensorShape value_shape;
    OP_REQUIRES_OK(context, resource->Peek(start, stop, &value_shape));

    Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, value_shape, &value_tensor));
    if (value_shape.dim_size(0) > 0) {
      OP_REQUIRES_OK(context, resource->Read(start, value_tensor));
    }
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
REGISTER_KERNEL_BUILDER(Name("IO>WAVReadableInit").Device(DEVICE_CPU),
                        WAVReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>WAVReadableSpec").Device(DEVICE_CPU),
                        WAVReadableSpecOp);
REGISTER_KERNEL_BUILDER(Name("IO>WAVReadableRead").Device(DEVICE_CPU),
                        WAVReadableReadOp);

}  // namespace data
}  // namespace tensorflow
