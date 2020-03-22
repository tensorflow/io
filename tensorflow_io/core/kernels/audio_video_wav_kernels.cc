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

#include "tensorflow_io/core/kernels/audio_kernels.h"

namespace tensorflow {
namespace data {
namespace {

// See http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
struct WAVHeader {
  char riff[4];           // RIFF Chunk ID: "RIFF"
  int32 riff_size;        // RIFF chunk size: 4 + n (file size - 8)
  char wave[4];           // WAVE ID: "WAVE"
  char fmt[4];            // fmt Chunk ID: "fmt "
  int32 fmt_size;         // fmt Chunk size: 16, 18, or 40
  int16 wFormatTag;       // Format code: WAVE_FORMAT_PCM (1) for PCM.
                          // WAVE_FORMAT_EXTENSIBLE (0xFFFE) for SubFormat
  int16 nChannels;        // Number of channels
  int32 nSamplesPerSec;   // Sampling rate
  int32 nAvgBytesPerSec;  // Data rate
  int16 nBlockAlign;      // Data block size (bytes)
  int16 wBitsPerSample;   // Bits per sample
};

struct ExtensionHeader {
  int16 cbSize;               // Size of the extension (0 or 22)
  int16 wValidBitsPerSample;  // Number of valid bits
  int32 dwChannelMask;        // Speaker position mask
  char SubFormat[16];         // GUID
};

struct DataHeader {
  char mark[4];
  int32 size;
};
Status ValidateWAVHeader(struct WAVHeader* header) {
  if (memcmp(header->riff, "RIFF", 4) != 0) {
    return errors::InvalidArgument("WAV file must starts with `RIFF`");
  }
  if (memcmp(header->wave, "WAVE", 4) != 0) {
    return errors::InvalidArgument("WAV file must contains riff type `WAVE`");
  }
  if (memcmp(header->fmt, "fmt ", 4) != 0) {
    return errors::InvalidArgument("WAV file must contains `fmt ` mark");
  }
  if (header->fmt_size != 16 && header->fmt_size != 18 &&
      header->fmt_size != 40) {
    return errors::InvalidArgument(
        "WAV file must have `fmt_size ` 16, 18, or 40, received: ",
        header->fmt_size);
  }
  if (header->wFormatTag != 1 &&
      header->wFormatTag != static_cast<int16>(0xFFFE)) {
    return errors::InvalidArgument(
        "WAV file must have `wFormatTag` 1 or 0xFFFE, received: ",
        header->wFormatTag);
  }
  if (header->nChannels <= 0) {
    return errors::InvalidArgument("WAV file have invalide channels: ",
                                   header->nChannels);
  }
  return Status::OK();
}

class WAVReadableResource : public AudioReadableResourceBase {
 public:
  WAVReadableResource(Env* env) : env_(env) {}
  ~WAVReadableResource() {}

  Status Init(const string& input, const void* optional_memory,
              const size_t optional_length) override {
    mutex_lock l(mu_);
    const string& filename = input;
    file_.reset(new SizedRandomAccessFile(env_, filename, optional_memory,
                                          optional_length));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    StringPiece result;
    TF_RETURN_IF_ERROR(
        file_->Read(0, sizeof(header_), &result, (char*)(&header_)));
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
      TF_RETURN_IF_ERROR(file_->Read(fmt_position, sizeof(header_) - 12,
                                     &result, (char*)(&header_) + 12));
      header_length_ = fmt_position + sizeof(header_) - 12;
    }

    TF_RETURN_IF_ERROR(ValidateWAVHeader(&header_));
    if (header_.riff_size + 8 != file_size_) {
      // corrupted file?
    }
    int64 filesize = header_.riff_size + 8;
    int64 position = header_length_ + header_.fmt_size - 16;

    int64 nSamples = 0;
    partitions_.clear();
    do {
      struct DataHeader head;
      TF_RETURN_IF_ERROR(
          file_->Read(position, sizeof(head), &result, (char*)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        // Data should be block aligned
        // bytes = nSamples * nBlockAlign
        if (head.size % header_.nBlockAlign != 0) {
          return errors::InvalidArgument("data chunk should be block aligned (",
                                         header_.nBlockAlign,
                                         "), received: ", head.size);
        }
        nSamples += head.size / header_.nBlockAlign;
        partitions_.emplace_back(nSamples);
        partitions_offset_.emplace_back(position);
      }
      position += head.size;
    } while (position < filesize);

    // Note: 8 bit is always 0-255 (uint8)
    switch (header_.wBitsPerSample) {
      case 8:
        dtype_ = DT_UINT8;
        break;
      case 16:
        dtype_ = DT_INT16;
        break;
      case 24:
        dtype_ = DT_INT32;
        break;
      default:
        return errors::InvalidArgument("unsupported wBitsPerSample: ",
                                       header_.wBitsPerSample);
    }

    shape_ = TensorShape({nSamples, header_.nChannels});

    rate_ = header_.nSamplesPerSec;

    return Status::OK();
  }

  Status Spec(TensorShape* shape, DataType* dtype, int32* rate) override {
    mutex_lock l(mu_);
    *shape = shape_;
    *dtype = dtype_;
    *rate = rate_;
    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop,
              std::function<Status(const TensorShape& shape, Tensor** value)>
                  allocate_func) override {
    mutex_lock l(mu_);
    int64 sample_stop =
        (stop < 0) ? (shape_.dim_size(0))
                   : (stop < shape_.dim_size(0) ? stop : shape_.dim_size(0));
    int64 sample_start = (start >= sample_stop) ? sample_stop : start;

    Tensor* value;
    TF_RETURN_IF_ERROR(allocate_func(
        TensorShape({sample_stop - sample_start, shape_.dim_size(1)}), &value));
    if (sample_stop == start) {
      return Status::OK();
    }

    const int64 channels = shape_.dim_size(1);

    int64 lower, upper, extra;
    TF_RETURN_IF_ERROR(
        PartitionsLookup(partitions_, start, stop, &lower, &upper, &extra));

    int64 base_offset = 0;
    char* base;
    switch (dtype_) {
      case DT_UINT8:
        base = (char*)(value->flat<uint8>().data());
        break;
      case DT_INT16:
        base = (char*)(value->flat<int16>().data());
        break;
      case DT_INT32:
        base = (char*)(value->flat<int32>().data());
        break;
      default:
        return errors::InvalidArgument("data type ", DataTypeString(dtype_),
                                       " not supported");
    }
    for (int64 i = lower; i < upper; i++) {
      int64 chunk_offset = 0;
      int64 chunk_length =
          (i == 0) ? (partitions_[i]) : (partitions_[i] - partitions_[i - 1]);
      // extra only applies to the first chunk
      if (i == lower) {
        chunk_offset = extra;
        chunk_length = chunk_length - extra;
      }
      // make sure copied chunk size is within the value shape
      if (base_offset + chunk_length > value->shape().dim_size(0)) {
        chunk_length = value->shape().dim_size(0) - base_offset;
      }
      if (chunk_length == 0) {
        continue;
      }

      int64 offset = partitions_offset_[i] + chunk_offset * header_.nBlockAlign;
      int64 length = chunk_length * header_.nBlockAlign;

      string buffer;
      buffer.resize(length);

      StringPiece result;
      TF_RETURN_IF_ERROR(file_->Read(offset, length, &result, &buffer[0]));

      switch (header_.wBitsPerSample) {
        case 8:
        case 16:
          memcpy(base + base_offset * header_.nBlockAlign, (char*)(&buffer[0]),
                 chunk_length * header_.nBlockAlign);
          break;
        case 24:
          for (int64 i = 0; i < chunk_length * channels; i++) {
            char* in_p = (char*)(&buffer[0]) + i * 3;
            char* out_p = base + base_offset * header_.nBlockAlign + i * 4;
            out_p[3] = in_p[2];
            out_p[2] = in_p[1];
            out_p[1] = in_p[0];
            out_p[0] = 0x00;
          }
          break;
        default:
          return errors::InvalidArgument(
              "unsupported wBitsPerSample and header.nBlockAlign: ",
              header_.wBitsPerSample, ", ", header_.nBlockAlign);
      }
      base_offset += chunk_length;
    }

    return Status::OK();
  }
  string DebugString() const override { return "WAVReadableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  DataType dtype_;
  TensorShape shape_;
  int64 rate_;

  struct WAVHeader header_;
  int64 header_length_;

  std::vector<int64> partitions_;
  std::vector<int64> partitions_offset_;
};

class AudioDecodeWAVOp : public OpKernel {
 public:
  explicit AudioDecodeWAVOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* shape_tensor;
    OP_REQUIRES_OK(context, context->input("shape", &shape_tensor));

    const string& input = input_tensor->scalar<tstring>()();

    std::unique_ptr<WAVReadableResource> resource(
        new WAVReadableResource(env_));
    OP_REQUIRES_OK(context,
                   resource->Init("memory", input.data(), input.size()));

    int32 rate;
    DataType dtype;
    TensorShape shape;
    OP_REQUIRES_OK(context, resource->Spec(&shape, &dtype, &rate));

    OP_REQUIRES(context, (dtype == context->expected_output_dtype(0)),
                errors::InvalidArgument(
                    "dtype mismatch: ", DataTypeString(dtype), " vs. ",
                    DataTypeString(context->expected_output_dtype(0))));

    PartialTensorShape provided_shape;
    OP_REQUIRES_OK(context, PartialTensorShape::MakePartialShape(
                                shape_tensor->flat<int64>().data(),
                                shape_tensor->NumElements(), &provided_shape));
    OP_REQUIRES(context, (provided_shape.IsCompatibleWith(shape)),
                errors::InvalidArgument(
                    "shape mismatch: ", provided_shape.DebugString(), " vs. ",
                    shape.DebugString()));

    OP_REQUIRES_OK(
        context,
        resource->Read(0, shape.dim_size(0),
                       [&](const TensorShape& shape, Tensor** value) -> Status {
                         TF_RETURN_IF_ERROR(
                             context->allocate_output(0, shape, value));
                         return Status::OK();
                       }));
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class AudioEncodeWAVOp : public OpKernel {
 public:
  explicit AudioEncodeWAVOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const Tensor* rate_tensor;
    OP_REQUIRES_OK(context, context->input("rate", &rate_tensor));

    const int64 channels = input_tensor->shape().dim_size(1);
    OP_REQUIRES(
        context, (channels == static_cast<int16>(channels)),
        errors::InvalidArgument("channels ", channels, " > max(int16)"));

    const int64 rate = rate_tensor->scalar<int64>()();
    OP_REQUIRES(context, (rate == static_cast<int32>(rate)),
                errors::InvalidArgument("rate ", rate, " > max(int32)"));

    int64 bytes_per_sample;
    char* input_base = nullptr;
    switch (input_tensor->dtype()) {
      case DT_UINT8:
        bytes_per_sample = 1;
        input_base = (char*)input_tensor->flat<uint8>().data();
        break;
      case DT_INT16:
        bytes_per_sample = 2;
        input_base = (char*)input_tensor->flat<int16>().data();
        break;
      case DT_INT32:
        bytes_per_sample = 3;
        input_base = (char*)input_tensor->flat<int32>().data();
        break;
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "data type ", DataTypeString(input_tensor->dtype()),
                        " not supported"));
    }
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &output_tensor));

    tstring& output = output_tensor->scalar<tstring>()();
    output.resize(sizeof(struct WAVHeader) + sizeof(struct DataHeader) +
                  input_tensor->NumElements() * bytes_per_sample);

    struct WAVHeader* header = (struct WAVHeader*)&output[0];
    struct DataHeader* data_header =
        (struct DataHeader*)&output[sizeof(struct WAVHeader)];
    char* output_base =
        (char*)&output[sizeof(struct WAVHeader) + sizeof(struct DataHeader)];

    // RIFF Chunk ID: "RIFF"
    memcpy(header->riff, "RIFF", 4);

    // placeholder RIFF chunk size: 4 + n (file size - 8)
    header->riff_size = output.size() - 8;

    // WAVE ID: "WAVE"
    memcpy(header->wave, "WAVE", 4);

    // fmt Chunk ID: "fmt "
    memcpy(header->fmt, "fmt ", 4);

    // fmt Chunk size: 16, 18, or 40
    header->fmt_size = 16;

    // Format code: WAVE_FORMAT_PCM (1) for PCM.
    // WAVE_FORMAT_EXTENSIBLE (0xFFFE) for SubFormat
    header->wFormatTag = 1;

    // Number of channels
    header->nChannels = channels;

    // Sampling rate
    header->nSamplesPerSec = rate;

    // Data rate
    // SampleRate * NumChannels * BitsPerSample/8 (BytesPerSample)
    header->nAvgBytesPerSec = rate * channels * bytes_per_sample;

    // Data block size (bytes)
    // NumChannels * BitsPerSample/8 (BytesPerSample)
    header->nBlockAlign = channels * bytes_per_sample;

    // Bits per sample (BytesPerSample * 8)
    header->wBitsPerSample = bytes_per_sample * 8;

    // Data Header
    memcpy(data_header->mark, "data", 4);
    data_header->size = input_tensor->NumElements() * bytes_per_sample;
    // Data Chunk
    switch (input_tensor->dtype()) {
      case DT_UINT8:
      case DT_INT16:
        memcpy(output_base, input_base,
               input_tensor->NumElements() * bytes_per_sample);
        break;
      case DT_INT32:
        for (int64 i = 0; i < input_tensor->NumElements(); i++) {
          char* in_p = input_base + i * 4;
          char* out_p = output_base + i * 3;
          out_p[2] = in_p[3];
          out_p[1] = in_p[2];
          out_p[0] = in_p[1];
        }
        break;
    }
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>AudioDecodeWAV").Device(DEVICE_CPU),
                        AudioDecodeWAVOp);
REGISTER_KERNEL_BUILDER(Name("IO>AudioEncodeWAV").Device(DEVICE_CPU),
                        AudioEncodeWAVOp);

}  // namespace

Status WAVReadableResourceInit(
    Env* env, const string& filename, const void* optional_memory,
    const size_t optional_length,
    std::unique_ptr<AudioReadableResourceBase>& resource) {
  resource.reset(new WAVReadableResource(env));
  Status status = resource->Init(filename, optional_memory, optional_length);
  if (!status.ok()) {
    resource.reset(nullptr);
  }
  return status;
}

}  // namespace data
}  // namespace tensorflow
