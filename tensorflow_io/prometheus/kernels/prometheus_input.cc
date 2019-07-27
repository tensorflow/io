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

#include "kernels/dataset_ops.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"

#include "go/prometheus.h"

namespace tensorflow {
namespace data {

class PrometheusState {
public:
  PrometheusState() : time_(0), offset_(0) {}

  int64 time_;
  int64 offset_;
};

class PrometheusInput: public StreamInput<PrometheusState> {
 public:
  Status ReadRecord(IteratorContext* ctx, std::unique_ptr<PrometheusState>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new PrometheusState());
      state.get()->time_ = time(NULL);
    }
    Tensor key_tensor(ctx->allocator({}), DT_INT64, {record_to_read});
    Tensor val_tensor(ctx->allocator({}), DT_DOUBLE, {record_to_read});
    GoSlice key_go = {key_tensor.flat<int64>().data(), record_to_read, record_to_read};
    GoSlice val_go = {val_tensor.flat<double>().data(), record_to_read, record_to_read};
    GoString endpoint_go = {endpoint().c_str(), static_cast<int64>(endpoint().size())};
    GoString query_go = {schema().c_str(), static_cast<int64>(schema().size())};

    GoInt returned = Query(endpoint_go, query_go, state.get()->time_, state.get()->offset_, key_go, val_go);
    if (returned < 0) {
        return errors::InvalidArgument("prometheus server error: ", returned);
    }
    if (returned > 0) {
      state.get()->offset_ += returned;
      *record_read = returned;
      if (*record_read < record_to_read) {
        Tensor key_tensor_final = key_tensor.Slice(0, *record_read);
        Tensor val_tensor_final = val_tensor.Slice(0, *record_read);
        out_tensors->emplace_back(std::move(key_tensor_final));
        out_tensors->emplace_back(std::move(val_tensor_final));
      } else {
        out_tensors->emplace_back(std::move(key_tensor));
        out_tensors->emplace_back(std::move(val_tensor));
      }
    }
    return Status::OK();
  }
  Status FromEndpoint(const string& endpoint) override {
    return Status::OK();
  }
  void EncodeAttributes(VariantTensorData* data) const override {
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    return true;
  }
 protected:
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(PrometheusInput, "tensorflow::data::PrometheusInput");

REGISTER_KERNEL_BUILDER(Name("PrometheusInput").Device(DEVICE_CPU),
                        StreamInputOp<PrometheusInput>);
REGISTER_KERNEL_BUILDER(Name("PrometheusDataset").Device(DEVICE_CPU),
                        StreamInputDatasetOp<PrometheusInput, PrometheusState>);
}  // namespace data
}  // namespace tensorflow
