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

namespace tensorflow {
namespace data {
namespace {

// NOTE: Both SizedRandomAccessFile and ArrowRandomAccessFile overlap
// with another PR. Will remove duplicate once PR merged

// Note: This SizedRandomAccessFile should only lives within Compute()
// of the kernel as buffer could be released by outside.
class SizedRandomAccessFile : public tensorflow::RandomAccessFile {
 public:
  SizedRandomAccessFile(Env* env, const string& filename, const string& optional_memory)
  : file_(nullptr)
  , size_status_(Status::OK())
  , size_(optional_memory.size())
  , buffer_(optional_memory) {
    if (size_ == 0) {
      size_status_ = env->GetFileSize(filename, &size_);
      if (size_status_.ok()) {
        size_status_ = env->NewRandomAccessFile(filename, &file_);
      }
    }
  }

  virtual ~SizedRandomAccessFile() {}
  Status Read(uint64 offset, size_t n, StringPiece* result, char* scratch) const override {
    if (file_.get() != nullptr) {
      return file_.get()->Read(offset, n, result, scratch);
    }
    size_t bytes_to_read = 0;
    if (offset < size_) {
      bytes_to_read = (offset + n < size_) ? n : (size_ - offset);
    }
    if (bytes_to_read > 0) {
      memcpy(scratch, &buffer_.data()[offset], bytes_to_read);
    }
    *result = StringPiece(scratch, bytes_to_read);
    if (bytes_to_read < n) {
      return errors::OutOfRange("EOF reached");
    }
    return Status::OK();
  }
  Status GetFileSize(uint64* size) {
    if (size_status_.ok()) {
      *size = size_;
    }
    return size_status_;
  }
 private:
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
  Status size_status_;
  uint64 size_;
  const string& buffer_;
};
struct WAVHeader {
  char riff[4]; // "RIFF"
  int32 size; // Size after (file size - 8)
  char wave[4]; // "WAVE"
  char fmt[4]; // "fmt "
  int32 fmt_size; // 16 for PCM
  int16 fmt_type; // 1 for PCM. 3 for IEEE Float
  int16 num_channels;
  int32 sample_rate;
  int32 byte_rate; // Number of bytes per second.
  int16 sample_alignment; // num_channels * Bytes Per Sample
  int16 bit_depth; // Number of bits per sample
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
  if (header->fmt_size != 16 && header->fmt_size != 18) {
    return errors::InvalidArgument("WAV file must have `fmt_size ` 16 or 18, received", header->fmt_size);
  }
  if (header->fmt_type != 1) {
    return errors::InvalidArgument("WAV file must have `fmt_type ` 1, received", header->fmt_type);
  }
  if (header->num_channels <= 0) {
    return errors::InvalidArgument("WAV file have invalide channels: ", header->num_channels);
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
    std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    StringPiece result;
    struct WAVHeader header;
    OP_REQUIRES_OK(context, file->Read(0, sizeof(header), &result, (char *)(&header)));

    OP_REQUIRES_OK(context, ValidateWAVHeader(&header));
    if (header.size + 8 != size) {
      // corrupted file?
    }
    int64 filesize = header.size + 8;

    int64 position = result.size();

    if (header.fmt_size == 18) {
      position += 2;
    }

    int64 bytes = 0;

    do {
      struct DataHeader head;
      OP_REQUIRES_OK(context, file->Read(position, sizeof(head), &result, (char *)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        bytes += head.size;
      }
      position += head.size;
    } while (position < filesize);

    string dtype;
    switch (header.bit_depth) {
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
      OP_REQUIRES(context, false, errors::InvalidArgument("unsupported bit_depth: ", header.bit_depth));
    }
    // bytes = NumSamples * NumChannels * BitsPerSample/8
    int64 num_samples = bytes / header.num_channels / (header.bit_depth / 8);

    Tensor* dtype_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &dtype_tensor));
    Tensor* shape_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({2}), &shape_tensor));
    Tensor* rate_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({}), &rate_tensor));

    dtype_tensor->scalar<string>()() = std::move(dtype);
    shape_tensor->flat<int64>()(0) = num_samples;
    shape_tensor->flat<int64>()(1) = header.num_channels;
    rate_tensor->scalar<int32>()() = header.sample_rate;
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

    const Tensor& start_tensor = context->input(1);
    int64 start = start_tensor.scalar<int64>()();

    const Tensor& count_tensor = context->input(2);
    int64 count = count_tensor.scalar<int64>()();

    const Tensor& memory_tensor = context->input(3);
    const string& memory = memory_tensor.scalar<string>()();

    std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    StringPiece result;
    struct WAVHeader header;
    OP_REQUIRES_OK(context, file->Read(0, sizeof(header), &result, (char *)(&header)));

    OP_REQUIRES_OK(context, ValidateWAVHeader(&header));
    if (header.size + 8 != size) {
      // corrupted file?
    }
    int64 filesize = header.size + 8;

    int64 position = result.size();

    if (header.fmt_size == 18) {
      position += 2;
    }

    int64 bytes = 0;

    do {
      struct DataHeader head;
      OP_REQUIRES_OK(context, file->Read(position, sizeof(head), &result, (char *)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        bytes += head.size;
      }
      position += head.size;
    } while (position < filesize);

    // bytes = NumSamples * NumChannels * BitsPerSample/8
    int64 num_samples = bytes / header.num_channels / (header.bit_depth / 8);

    if (start > num_samples) {
      start = num_samples;
    }
    if (count < 0) {
      count = num_samples - start;
    }

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({count, header.num_channels}), &output_tensor));

    int64 read_offset = 0;
    int64 start_bytes = start * header.num_channels * (header.bit_depth / 8);
    int64 final_bytes = start_bytes + count * header.num_channels * (header.bit_depth / 8);

    bytes = 0;

    position = sizeof(header) + ((header.fmt_size == 18) ? 2 : 0);
    do {
      struct DataHeader head;
      OP_REQUIRES_OK(context, file->Read(position, sizeof(head), &result, (char *)(&head)));
      position += result.size();
      if (memcmp(head.mark, "data", 4) == 0) {
        // only read if start_bytes and final_bytes within range:
        if (start_bytes < bytes + head.size && final_bytes > bytes) {
          int64 read_start_bytes = (start_bytes < bytes) ? bytes : start_bytes;
          int64 read_final_bytes = (final_bytes < bytes + head.size) ? final_bytes : (bytes + head.size);
          string buffer;
          buffer.resize(read_final_bytes - read_start_bytes);
          OP_REQUIRES_OK(context, file->Read(position + read_start_bytes - bytes, (read_final_bytes - read_start_bytes), &result, &buffer[0]));

          switch (header.bit_depth) {
          case 8:
            memcpy((char *)(output_tensor->flat<int8>().data()) + read_offset, &buffer[0], (read_final_bytes - read_start_bytes));
            read_offset += (read_final_bytes - read_start_bytes);
            break;
          case 16:
            memcpy((char *)(output_tensor->flat<int16>().data()) + read_offset, &buffer[0], (read_final_bytes - read_start_bytes));
            read_offset += (read_final_bytes - read_start_bytes);
            break;
          default:
            OP_REQUIRES(context, false, errors::InvalidArgument("unsupported bit_depth: ", header.bit_depth));
          }
        }
        bytes += head.size;
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
