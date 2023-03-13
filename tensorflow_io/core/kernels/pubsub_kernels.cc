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

#include <grpc++/grpc++.h>
// Inclusion of googleapi related grpc headers, e.g., pubsub.grpc.pb.h
// will cause Windows build failures due to the conflict of `OPTIONAL`
// definition. The following is needed for Windows.
#if defined(_MSC_VER)
#include <Windows.h>
#undef OPTIONAL
#endif
#include "absl/time/clock.h"
#include "google/pubsub/v1/pubsub.grpc.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {
namespace data {
namespace {

using google::pubsub::v1::AcknowledgeRequest;
using google::pubsub::v1::PullRequest;
using google::pubsub::v1::PullResponse;
using google::pubsub::v1::Subscriber;
using grpc::ClientContext;

class PubSubReadableResource : public ResourceBase {
 public:
  PubSubReadableResource(Env* env) : env_(env) {}
  ~PubSubReadableResource() {}

  Status Init(const string& input, const std::vector<string>& metadata) {
    mutex_lock l(mu_);

    endpoint_ = "";
    subscription_ = input;
    timeout_ = 10 * 1000;
    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find("endpoint=") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid configuration: ",
                                         metadata[i]);
        }
        endpoint_ = parts[1];
      } else if (metadata[i].find("timeout=") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2 || !strings::safe_strto64(parts[1], &timeout_)) {
          return errors::InvalidArgument("invalid configuration: ",
                                         metadata[i]);
        }
      }
    }
    string endpoint = endpoint_;
    auto creds = grpc::GoogleDefaultCredentials();
    if (endpoint_.find("http://") == 0) {
      endpoint = endpoint_.substr(7);
      creds = grpc::InsecureChannelCredentials();
    } else if (endpoint_.find("https://") == 0) {
      // https://pubsub.googleapis.com
      endpoint = endpoint_.substr(8);
    }
    stub_ = Subscriber::NewStub(grpc::CreateChannel(endpoint, creds));

    return OkStatus();
  }
  Status Read(std::function<Status(const TensorShape& shape, Tensor** id_tensor,
                                   Tensor** data_tensor, Tensor** time_tensor)>
                  allocate_func) {
    mutex_lock l(mu_);
    if (stub_.get() == nullptr) {
      return errors::OutOfRange("EOF reached");
    }
    ClientContext context;
    if (timeout_ > 0) {
      std::chrono::system_clock::time_point deadline =
          std::chrono::system_clock::now() +
          std::chrono::milliseconds(timeout_);
      context.set_deadline(deadline);
    }
    Tensor* id_tensor;
    Tensor* data_tensor;
    Tensor* time_tensor;
    while (true) {
      PullRequest request;
      request.set_subscription(subscription_);
      request.set_max_messages(1);
      PullResponse response;
      auto status = stub_->Pull(&context, request, &response);
      if (!status.ok()) {
        return errors::Internal("Failed to receive message: ",
                                status.error_message());
      }
      if (response.received_messages().size() == 0 && timeout_ > 0) {
        // break subscription if there is a timeout, and no message.
        TF_RETURN_IF_ERROR(allocate_func(TensorShape({0}), &id_tensor,
                                         &data_tensor, &time_tensor));
        stub_.reset(nullptr);
        return OkStatus();
      }
      if (response.received_messages().size() != 0) {
        TF_RETURN_IF_ERROR(allocate_func(TensorShape({1}), &id_tensor,
                                         &data_tensor, &time_tensor));
        id_tensor->scalar<tstring>()() =
            std::string((response.received_messages(0).message().message_id()));
        data_tensor->scalar<tstring>()() =
            std::string((response.received_messages(0).message().data()));
        time_tensor->scalar<int64>()() =
            response.received_messages(0).message().publish_time().seconds() *
                1000 +
            response.received_messages(0).message().publish_time().nanos() /
                1000000;

        // Acknowledge
        AcknowledgeRequest acknowledge;
        acknowledge.add_ack_ids(response.received_messages(0).ack_id());
        acknowledge.set_subscription(subscription_);
        google::protobuf::Empty empty;
        ClientContext ack_context;
        status = stub_->Acknowledge(&ack_context, acknowledge, &empty);

        return OkStatus();
      }
    }
    return OkStatus();
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return "PubSubReadableResource";
  }

 protected:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  string subscription_ TF_GUARDED_BY(mu_);
  string endpoint_ TF_GUARDED_BY(mu_);
  int64 timeout_ TF_GUARDED_BY(mu_);
  std::unique_ptr<Subscriber::Stub> stub_ TF_GUARDED_BY(mu_);
  string message_id_ TF_GUARDED_BY(mu_);
  string message_data_ TF_GUARDED_BY(mu_);
  int64 message_time_ TF_GUARDED_BY(mu_);
};

class PubSubReadableInitOp : public ResourceOpKernel<PubSubReadableResource> {
 public:
  explicit PubSubReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<PubSubReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<PubSubReadableResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<tstring>()();

    std::vector<string> metadata;
    const Tensor* metadata_tensor;
    OP_REQUIRES_OK(context, context->input("metadata", &metadata_tensor));
    for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
      metadata.push_back(metadata_tensor->flat<tstring>()(i));
    }

    OP_REQUIRES_OK(context, resource_->Init(input, metadata));
  }
  Status CreateResource(PubSubReadableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new PubSubReadableResource(env_);
    return OkStatus();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class PubSubReadableReadOp : public OpKernel {
 public:
  explicit PubSubReadableReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    PubSubReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    OP_REQUIRES_OK(
        context, resource->Read([&](const TensorShape& shape,
                                    Tensor** id_tensor, Tensor** data_tensor,
                                    Tensor** time_tensor) -> Status {
          TF_RETURN_IF_ERROR(context->allocate_output(0, shape, id_tensor));
          TF_RETURN_IF_ERROR(context->allocate_output(1, shape, data_tensor));
          TF_RETURN_IF_ERROR(context->allocate_output(2, shape, time_tensor));
          return OkStatus();
        }));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};
REGISTER_KERNEL_BUILDER(Name("IO>PubSubReadableInit").Device(DEVICE_CPU),
                        PubSubReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>PubSubReadableRead").Device(DEVICE_CPU),
                        PubSubReadableReadOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
