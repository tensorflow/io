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
#include "tensorflow/core/lib/io/random_inputstream.h"

namespace tensorflow {
namespace data {

class StorageReadOp : public OpKernel {
 public:
  explicit StorageReadOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const Tensor& offset_tensor = context->input(1);
    const Tensor& length_tensor = context->input(2);

    std::vector<int64> offset;
    // Populate offset if needed
    if (offset_tensor.NumElements() > 1) {
      // TODO: We should be able to broadcast
      OP_REQUIRES(context, (input_tensor.NumElements() == offset_tensor.NumElements()), errors::InvalidArgument("input and offset does not match: ", input_tensor.NumElements(), " vs. ", offset_tensor.NumElements()));
      offset.reserve(input_tensor.NumElements());
      for (int64 i = 0; i < input_tensor.NumElements(); i++) {
        offset[i] = offset_tensor.flat<int64>()(i);
      }
    } else {
      const int64 value = (offset_tensor.NumElements() == 0) ? (0) : (offset_tensor.scalar<int64>()());
      offset.resize(input_tensor.NumElements(), value);
    }

    std::vector<int64> length;
    // Populate length if needed
    if (length_tensor.NumElements() > 1) {
      // TODO: We should be able to broadcast
      OP_REQUIRES(context, (input_tensor.NumElements() == length_tensor.NumElements()), errors::InvalidArgument("input and length does not match: ", input_tensor.NumElements(), " vs. ", length_tensor.NumElements()));
      length.reserve(input_tensor.NumElements());
      for (int64 i = 0; i < input_tensor.NumElements(); i++) {
        length[i] = length_tensor.flat<int64>()(i);
      }
    } else {
      const int64 value = (length_tensor.NumElements() == 0) ? (-1) : (length_tensor.scalar<int64>()());
      length.resize(input_tensor.NumElements(), value);
    }

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    for (int64 i = 0; i < input_tensor.NumElements(); i++) {
      // TODO: multi-thread?
      uint64 size = 0;
      OP_REQUIRES_OK(context, env_->GetFileSize(input_tensor.flat<string>()(i), &size));

      std::unique_ptr<RandomAccessFile> file;
      OP_REQUIRES_OK(context, env_->NewRandomAccessFile(input_tensor.flat<string>()(i), &file));

      size_t n = (length[i] >= 0) ? (static_cast<size_t>(length[i])) : (size < offset[i] ? 0 : static_cast<size_t>(size - offset[i]));
      string buffer;
      StringPiece result;
      buffer.resize(n);
      Status status = file->Read(offset[i], n, &result, static_cast<char*>(&buffer[0]));
      OP_REQUIRES(context, (status.ok() || errors::IsOutOfRange(status)), status);
      output_tensor->flat<string>()(i) = buffer;
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class StorageSizeOp : public OpKernel {
 public:
  explicit StorageSizeOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    for (int64 i = 0; i < input_tensor.NumElements(); i++) {
      // TODO: multi-thread?
      uint64 size = 0;
      OP_REQUIRES_OK(context, env_->GetFileSize(input_tensor.flat<string>()(i), &size));
      output_tensor->flat<int64>()(i) = size;
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("StorageRead").Device(DEVICE_CPU),
                        StorageReadOp);
REGISTER_KERNEL_BUILDER(Name("StorageSize").Device(DEVICE_CPU),
                        StorageSizeOp);
}  // namespace data
}  // namespace tensorflow
