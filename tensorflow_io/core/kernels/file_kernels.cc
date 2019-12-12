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
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow_io/core/kernels/io_stream.h"

namespace tensorflow {
namespace data {
namespace {

class FileInfoOp : public OpKernel {
 public:
  explicit FileInfoOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<string>()();

    uint64 size;
    OP_REQUIRES_OK(context, env_->GetFileSize(input, &size));

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    OP_REQUIRES_OK(context, env_->NewRandomAccessFile(input, &file));

    Tensor* length_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &length_tensor));
    length_tensor->scalar<int64>()() = size;

    Tensor* compression_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                     &compression_tensor));

    StringPiece result;
    char buffer[10] = {0};
    Status status = file->Read(0, 10, &result, buffer);
    if (!status.ok() || result.size() != 10) {
      return;
    }
    // deflation- third byte must be 0x08.
    if (memcmp(buffer, "\x1F\x8B\x08", 3) != 0) {
      return;
    }
    // No reserved flags set.
    if ((buffer[3] & 0xE0) != 0) {
      return;
    }
    compression_tensor->scalar<string>()() = "GZIP";
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class FileReadOp : public OpKernel {
 public:
  explicit FileReadOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<string>()();

    const Tensor* offset_tensor;
    OP_REQUIRES_OK(context, context->input("offset", &offset_tensor));
    const int64 offset = offset_tensor->scalar<int64>()();

    const Tensor* length_tensor;
    OP_REQUIRES_OK(context, context->input("length", &length_tensor));
    const int64 length = length_tensor->scalar<int64>()();

    const Tensor* compression_tensor;
    OP_REQUIRES_OK(context, context->input("compression", &compression_tensor));
    const string& compression = compression_tensor->scalar<string>()();

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    OP_REQUIRES_OK(context, env_->NewRandomAccessFile(input, &file));

    std::unique_ptr<tensorflow::io::RandomAccessInputStream> input_stream(
        new tensorflow::io::RandomAccessInputStream(file.get()));

    tensorflow::io::InputStreamInterface* stream = input_stream.get();

    std::unique_ptr<tensorflow::io::ZlibInputStream> zlib_stream;
    if (compression == "GZIP") {
      zlib_stream.reset(new tensorflow::io::ZlibInputStream(
          input_stream.get(), 256 * 1024, 256 * 1024,
          tensorflow::io::ZlibCompressionOptions::GZIP()));
      stream = zlib_stream.get();
    }

    OP_REQUIRES_OK(context, stream->SkipNBytes(offset));

    string value;
    OP_REQUIRES_OK(context, stream->ReadNBytes(length, &value));

    Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &value_tensor));
    value_tensor->scalar<string>()() = value;
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>FileInfo").Device(DEVICE_CPU), FileInfoOp);
REGISTER_KERNEL_BUILDER(Name("IO>FileRead").Device(DEVICE_CPU), FileReadOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
