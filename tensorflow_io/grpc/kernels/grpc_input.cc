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
#include "endpoint.grpc.pb.h"
#include <grpc++/grpc++.h>

namespace tensorflow {
namespace data {

class GRPCInputState {
public:
  GRPCInputState(const string& endpoint) : offset_(0) {
    stub_ = GRPCEndpoint::NewStub(grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials()));
  }
  int64 offset_;
  std::unique_ptr<GRPCEndpoint::Stub> stub_;
};

class GRPCInput: public StreamInput<GRPCInputState> {
 public:
  Status ReadRecord(IteratorContext* ctx, std::unique_ptr<GRPCInputState>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new GRPCInputState(endpoint_));
    }
    Request request;
    request.set_offset(state.get()->offset_);
    request.set_length(record_to_read);
    Response response;
    grpc::ClientContext context;
    grpc::Status status = state.get()->stub_.get()->ReadRecord(&context, request, &response);
    if (!status.ok()) {
      return errors::InvalidArgument("unable to fetch data from grpc (", status.error_code(), "): ", status.error_message());
    }
    TensorProto record;
    response.record().UnpackTo(&record);
    Tensor value_tensor;
    value_tensor.FromProto(ctx->allocator({}), record);
    out_tensors->emplace_back(std::move(value_tensor));

    *record_read = value_tensor.dim_size(0);
    state.get()->offset_ += *record_read;

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

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(GRPCInput, "tensorflow::data::GRPCInput");

REGISTER_KERNEL_BUILDER(Name("IO>GRPCInput").Device(DEVICE_CPU),
                        StreamInputOp<GRPCInput>);
REGISTER_KERNEL_BUILDER(Name("IO>GRPCDataset").Device(DEVICE_CPU),
                        StreamInputDatasetOp<GRPCInput, GRPCInputState>);
}  // namespace data
}  // namespace tensorflow
