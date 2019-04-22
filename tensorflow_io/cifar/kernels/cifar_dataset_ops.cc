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
  Status ReadRecord(io::InputStreamInterface& s, IteratorContext* ctx, std::unique_ptr<int64>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new int64(0));
    }
    string buffer;
    TF_RETURN_IF_ERROR(ReadInputStream(s, 3073, record_to_read, &buffer, record_read));
    (*(state.get())) += *record_read;
    if (*record_read > 0) {
      Tensor label_tensor(ctx->allocator({}), DT_UINT8, {*record_read});
      Tensor value_tensor(ctx->allocator({}), DT_UINT8, {*record_read, 3, 32, 32});
      for (int64 i = 0; i < (*record_read); i++) {
        // Memory alignment?
        memcpy(&(label_tensor.flat<uint8>().data()[i]), &(buffer.data()[i * 3073 + 0]), 1);
        memcpy(&(value_tensor.flat<uint8>().data()[i * 3072]), &(buffer.data()[i * 3073 + 1]), 3072);
      }
      out_tensors->emplace_back(std::move(label_tensor));
      out_tensors->emplace_back(std::move(value_tensor));
    }
    return Status::OK();
  }
  Status FromStream(io::InputStreamInterface& s) override {
    return Status::OK();
  }
  void EncodeAttributes(VariantTensorData* data) const override {
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    return true;
  }
 protected:
};

class CIFAR100Input: public DataInput<int64> {
 public:
  Status ReadRecord(io::InputStreamInterface& s, IteratorContext* ctx, std::unique_ptr<int64>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new int64(0));
    }
    string buffer;
    TF_RETURN_IF_ERROR(ReadInputStream(s, 3074, record_to_read, &buffer, record_read));
    (*(state.get())) += *record_read;
    if (*record_read > 0) {
      Tensor coarse_tensor(ctx->allocator({}), DT_UINT8, {*record_read});
      Tensor fine_tensor(ctx->allocator({}), DT_UINT8, {*record_read});
      Tensor value_tensor(ctx->allocator({}), DT_UINT8, {*record_read, 3, 32, 32});
      for (int64 i = 0; i < (*record_read); i++) {
        // Memory alignment?
        memcpy(&(coarse_tensor.flat<uint8>().data()[i]), &(buffer.data()[i * 3074 + 0]), 1);
        memcpy(&(fine_tensor.flat<uint8>().data()[i]), &(buffer.data()[i * 3074 + 1]), 1);
        memcpy(&(value_tensor.flat<uint8>().data()[i * 3072]), &(buffer.data()[i * 3074 + 2]), 3072);
      }
      out_tensors->emplace_back(std::move(coarse_tensor));
      out_tensors->emplace_back(std::move(fine_tensor));
      out_tensors->emplace_back(std::move(value_tensor));
    }
    return Status::OK();
  }
  Status FromStream(io::InputStreamInterface& s) override {
    return Status::OK();
  }
  void EncodeAttributes(VariantTensorData* data) const override {
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    return true;
  }
 protected:
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
