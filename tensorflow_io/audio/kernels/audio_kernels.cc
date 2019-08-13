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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_io/core/kernels/stream.h"

namespace tensorflow {
namespace data {
namespace {

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

class ListWAVInfoOp : public OpKernel {
 public:
  explicit ListWAVInfoOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string filename = filename_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(1);
    const string& memory = memory_tensor.scalar<string>()();
    std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory.data(), memory.size()));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    StringPiece result;
    struct WAVHeader header;
    OP_REQUIRES_OK(context, file->Read(0, sizeof(header), &result, (char *)(&header)));

    OP_REQUIRES_OK(context, ValidateWAVHeader(&header));
    if (header.riff_size + 8 != size) {
      // corrupted file?
    }
    int64 filesize = header.riff_size + 8;

    int64 position = result.size();

    if (header.fmt_size != 16) {
      position += header.fmt_size - 16;
    }

    int64 nSamples = 0;
    do {
      struct DataHeader head;
      OP_REQUIRES_OK(context, file->Read(position, sizeof(head), &result, (char *)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        // Data should be block aligned
        // bytes = nSamples * nBlockAlign
        OP_REQUIRES(context, (head.size % header.nBlockAlign == 0), errors::InvalidArgument("data chunk should be block aligned (", header.nBlockAlign, "), received: ", head.size));
        nSamples += head.size / header.nBlockAlign;
      }
      position += head.size;
    } while (position < filesize);

    string dtype;
    switch (header.wBitsPerSample) {
    case 8:
      dtype = "int8";
      break;
    case 16:
      dtype = "int16";
      break;
    case 24:
      dtype = "int32";
      break;
    default:
      OP_REQUIRES(context, false, errors::InvalidArgument("unsupported wBitsPerSample: ", header.wBitsPerSample));
    }

    Tensor* dtype_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &dtype_tensor));
    Tensor* shape_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({2}), &shape_tensor));
    Tensor* rate_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({}), &rate_tensor));

    dtype_tensor->scalar<string>()() = std::move(dtype);
    shape_tensor->flat<int64>()(0) = nSamples;
    shape_tensor->flat<int64>()(1) = header.nChannels;
    rate_tensor->scalar<int32>()() = header.nSamplesPerSec;
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class ReadWAVOp : public OpKernel {
 public:
  explicit ReadWAVOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string& filename = filename_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(1);
    const string& memory = memory_tensor.scalar<string>()();

    const Tensor& start_tensor = context->input(2);
    const int64 start = start_tensor.scalar<int64>()();

    const Tensor& stop_tensor = context->input(3);
    const int64 stop = stop_tensor.scalar<int64>()();

    std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory.data(), memory.size()));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    StringPiece result;
    struct WAVHeader header;
    OP_REQUIRES_OK(context, file->Read(0, sizeof(header), &result, (char *)(&header)));

    OP_REQUIRES_OK(context, ValidateWAVHeader(&header));
    if (header.riff_size + 8 != size) {
      // corrupted file?
    }
    int64 filesize = header.riff_size + 8;

    int64 position = result.size();
    if (header.fmt_size != 16) {
      position += header.fmt_size - 16;
    }

    int64 nSamples = 0;
    do {
      struct DataHeader head;
      OP_REQUIRES_OK(context, file->Read(position, sizeof(head), &result, (char *)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        // Data should be block aligned
        // bytes = nSamples * nBlockAlign
        OP_REQUIRES(context, (head.size % header.nBlockAlign == 0), errors::InvalidArgument("data chunk should be block aligned (", header.nBlockAlign, "), received: ", head.size));
        nSamples += head.size / header.nBlockAlign;
      }
      position += head.size;
    } while (position < filesize);


    int64 sample_start = start;
    int64 sample_stop = stop;
    if (sample_start > nSamples) {
      sample_start = nSamples;
    }
    if (sample_stop < 0) {
      sample_stop = nSamples;
    }
    if (sample_stop < sample_start) {
      sample_stop = sample_start;
    }
    

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({sample_stop - sample_start, header.nChannels}), &output_tensor));

    int64 sample_offset = 0;

    position = sizeof(header) + header.fmt_size - 16;
    do {
      struct DataHeader head;
      OP_REQUIRES_OK(context, file->Read(position, sizeof(head), &result, (char *)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        // Already checked the alignment
        int64 block_sample_start = sample_offset;
        int64 block_sample_stop = sample_offset + head.size / header.nBlockAlign;
        // only read if block_sample_start and block_sample_stop within range
        if (sample_start < block_sample_stop && sample_stop > block_sample_start) {
          int64 read_sample_start = (block_sample_start > sample_start ? block_sample_start : sample_start);
          int64 read_sample_stop = (block_sample_stop < sample_stop ? block_sample_stop : sample_stop);
          int64 read_bytes_start = position + (read_sample_start - block_sample_start) * header.nBlockAlign;
          int64 read_bytes_stop = position + (read_sample_stop - block_sample_start) * header.nBlockAlign;
          string buffer;
          buffer.resize(read_bytes_stop - read_bytes_start);
          OP_REQUIRES_OK(context, file->Read(read_bytes_start, read_bytes_stop - read_bytes_start, &result, &buffer[0]));
          switch (header.wBitsPerSample) {
          case 8:
            OP_REQUIRES(context, (header.wBitsPerSample * header.nChannels == header.nBlockAlign * 8), errors::InvalidArgument("unsupported wBitsPerSample and header.nBlockAlign: ", header.wBitsPerSample, ", ", header.nBlockAlign));
            memcpy((char *)(output_tensor->flat<int8>().data()) + ((read_sample_start - sample_start) * header.nBlockAlign), &buffer[0], (read_bytes_stop - read_bytes_start));
            break;
          case 16:
            OP_REQUIRES(context, (header.wBitsPerSample * header.nChannels == header.nBlockAlign * 8), errors::InvalidArgument("unsupported wBitsPerSample and header.nBlockAlign: ", header.wBitsPerSample, ", ", header.nBlockAlign));
            memcpy((char *)(output_tensor->flat<int16>().data()) + ((read_sample_start - sample_start) * header.nBlockAlign), &buffer[0], (read_bytes_stop - read_bytes_start));
            break;
          case 24:
            // NOTE: The conversion is from signed integer 24 to signed integer 32 (left shift 8 bits)
            OP_REQUIRES(context, (header.wBitsPerSample * header.nChannels == header.nBlockAlign * 8), errors::InvalidArgument("unsupported wBitsPerSample and header.nBlockAlign: ", header.wBitsPerSample, ", ", header.nBlockAlign));
            for (int64 i = read_sample_start; i < read_sample_stop; i++) {
              for (int64 j = 0; j < header.nChannels; j++) {
                char *data_p = (char *)(output_tensor->flat<int32>().data() + ((i - sample_start) * header.nChannels + j));
                char *read_p = (char *)(&buffer[((i - read_sample_start) * header.nBlockAlign)]) + 3 * j;
                data_p[3] = read_p[2];
                data_p[2] = read_p[1];
                data_p[1] = read_p[0];
                data_p[0] = 0x00;
              }
            }
            break;
          default:
            OP_REQUIRES(context, false, errors::InvalidArgument("unsupported wBitsPerSample and header.nBlockAlign: ", header.wBitsPerSample, ", ", header.nBlockAlign));
          }
        }
        sample_offset = block_sample_stop;
      }
      position += head.size;
    } while (position < filesize);
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("ListWAVInfo").Device(DEVICE_CPU),
                        ListWAVInfoOp);
REGISTER_KERNEL_BUILDER(Name("ReadWAV").Device(DEVICE_CPU),
                        ReadWAVOp);


}  // namespace
}  // namespace data
}  // namespace tensorflow
