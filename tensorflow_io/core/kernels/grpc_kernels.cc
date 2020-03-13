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

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow_io/core/grpc/endpoint.grpc.pb.h"

#include <grpc++/grpc++.h>

namespace tensorflow {
namespace data {
namespace {

class GRPCReadableResource : public ResourceBase {
 public:
  GRPCReadableResource(Env* env) : env_(env) {}
  ~GRPCReadableResource() {}

  Status Init(const string& input) {
    mutex_lock l(mu_);
    endpoint_ = input;
    stub_ = GRPCEndpoint::NewStub(
        grpc::CreateChannel(endpoint_, grpc::InsecureChannelCredentials()));
    return Status::OK();
  }
  Status Read(const int64 start, const TensorShape& shape,
              std::function<Status(const TensorShape& shape, Tensor** value)>
                  allocate_func) {
    mutex_lock l(mu_);

    Tensor* value;
    TF_RETURN_IF_ERROR(allocate_func(shape, &value));
    if (shape.dim_size(0) == 0) {
      return Status::OK();
    }

    Request request;
    request.set_offset(start);
    request.set_length(shape.dim_size(0));
    Response response;
    grpc::ClientContext context;
    grpc::Status status = stub_->ReadRecord(&context, request, &response);
    if (!status.ok()) {
      return errors::InvalidArgument("unable to fetch data from grpc (",
                                     status.error_code(),
                                     "): ", status.error_message());
    }
    TensorProto record;
    response.record().UnpackTo(&record);

    if (!value->FromProto(record)) {
      return errors::InvalidArgument("unable to fill tensor");
    }

    return Status::OK();
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return "GRPCReadableResource";
  }

 protected:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  string endpoint_ GUARDED_BY(mu_);
  std::unique_ptr<GRPCEndpoint::Stub> stub_ GUARDED_BY(mu_);
};

class GRPCReadableInitOp : public ResourceOpKernel<GRPCReadableResource> {
 public:
  explicit GRPCReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<GRPCReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<GRPCReadableResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    string input = input_tensor->scalar<tstring>()();

    OP_REQUIRES_OK(context, resource_->Init(input));
  }
  Status CreateResource(GRPCReadableResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new GRPCReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class GRPCReadableReadOp : public OpKernel {
 public:
  explicit GRPCReadableReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    GRPCReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    const int64 start = start_tensor->scalar<int64>()();

    const Tensor* shape_tensor;
    OP_REQUIRES_OK(context, context->input("shape", &shape_tensor));
    TensorShape shape(shape_tensor->flat<int64>());

    OP_REQUIRES_OK(
        context,
        resource->Read(start, shape,
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
REGISTER_KERNEL_BUILDER(Name("IO>GRPCReadableInit").Device(DEVICE_CPU),
                        GRPCReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>GRPCReadableRead").Device(DEVICE_CPU),
                        GRPCReadableReadOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
