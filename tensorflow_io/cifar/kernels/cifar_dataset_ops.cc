/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <archive.h>
#include <archive_entry.h>

#define EIGEN_USE_THREADS

#include "kernels/dataset_ops.h"

namespace tensorflow {
namespace data {
namespace {
class CIFAR10Input: public DataInput<int64> {
 public:
  Status ReadRecord(io::InputStreamInterface& s, IteratorContext* ctx, std::unique_ptr<int64>& state, int64* returned, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new int64(0));
    }
    string buffer;
    TF_RETURN_IF_ERROR(ReadInputStream(s, 3073, 1, &buffer , returned));
    (*(state.get())) += *returned;
    if (*returned == 1) {
      Tensor label_tensor(ctx->allocator({}), DT_UINT8, {});
      label_tensor.scalar<uint8>()() = *((uint8 *)(&(buffer.data()[0])));
      out_tensors->emplace_back(std::move(label_tensor));

      Tensor value_tensor(ctx->allocator({}), DT_UINT8, {3, 32, 32});
      memcpy(value_tensor.flat<uint8>().data(), &(buffer.data()[1]), 3072);
      out_tensors->emplace_back(std::move(value_tensor));
    }
    return Status::OK();
  }
  Status FromStream(io::InputStreamInterface& s) override {
    size_ = 0;
    Status status = s.SkipNBytes(3073);
    while (status.ok()) {
      size_ += 1;
      status = s.SkipNBytes(3073);
    }
    if (status != errors::OutOfRange("EOF reached")) {
      return status;
    }
    return Status::OK();
  }
  void EncodeAttributes(VariantTensorData* data) const override {
    data->tensors_.emplace_back(Tensor(DT_INT64, TensorShape({})));
    data->tensors_.back().scalar<int64>()() = size_;
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    size_t size_index = data.tensors().size() - 1;
    size_ = data.tensors(size_index).scalar<int64>()();
    return true;
  }
 protected:
  int64 size_;
};

class CIFAR100Input: public DataInput<int64> {
 public:
  Status ReadRecord(io::InputStreamInterface& s, IteratorContext* ctx, std::unique_ptr<int64>& state, int64* returned, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new int64(0));
    }
    string buffer;
    TF_RETURN_IF_ERROR(ReadInputStream(s, 3074, 1, &buffer , returned));
    (*(state.get())) += *returned;
    if (*returned == 1) {
      Tensor coarse_tensor(ctx->allocator({}), DT_UINT8, {});
      coarse_tensor.scalar<uint8>()() = *((uint8 *)(&(buffer.data()[0])));
      out_tensors->emplace_back(std::move(coarse_tensor));

      Tensor fine_tensor(ctx->allocator({}), DT_UINT8, {});
      fine_tensor.scalar<uint8>()() = *((uint8 *)(&(buffer.data()[1])));
      out_tensors->emplace_back(std::move(fine_tensor));

      Tensor value_tensor(ctx->allocator({}), DT_UINT8, {3, 32, 32});
      memcpy(value_tensor.flat<uint8>().data(), &(buffer.data()[2]), 3072);
      out_tensors->emplace_back(std::move(value_tensor));
    }
    return Status::OK();
  }
  Status FromStream(io::InputStreamInterface& s) override {
    size_ = 0;
    Status status = s.SkipNBytes(3074);
    while (status.ok()) {
      size_ += 1;
      status = s.SkipNBytes(3074);
    }
    if (status != errors::OutOfRange("EOF reached")) {
      return status;
    }
    return Status::OK();
  }
  void EncodeAttributes(VariantTensorData* data) const override {
    data->tensors_.emplace_back(Tensor(DT_INT64, TensorShape({})));
    data->tensors_.back().scalar<int64>()() = size_;
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    size_t size_index = data.tensors().size() - 1;
    size_ = data.tensors(size_index).scalar<int64>()();
    return true;
  }
 protected:
  int64 size_;
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(CIFAR10Input, "tensorflow::CIFAR10Input");
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(CIFAR100Input, "tensorflow::CIFAR100Input");

REGISTER_KERNEL_BUILDER(Name("CIFAR10Input").Device(DEVICE_CPU),
                        DataInputOp<CIFAR10Input>);
REGISTER_KERNEL_BUILDER(Name("CIFAR100Input").Device(DEVICE_CPU),
                        DataInputOp<CIFAR100Input>);
REGISTER_KERNEL_BUILDER(Name("CIFAR10Dataset").Device(DEVICE_CPU),
                        InputDatasetOp<CIFAR10Input, int64>);
REGISTER_KERNEL_BUILDER(Name("CIFAR100Dataset").Device(DEVICE_CPU),
                        InputDatasetOp<CIFAR100Input, int64>);
}  // namespace
}  // namespace data
}  // namespace tensorflow
