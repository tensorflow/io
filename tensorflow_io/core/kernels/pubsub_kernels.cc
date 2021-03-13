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

#include "google/cloud/pubsub/subscriber.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/env_time.h"

namespace tensorflow {
namespace data {
namespace {

namespace pubsub = google::cloud::pubsub;

class PubSubReadableResource : public ResourceBase {
 public:
  PubSubReadableResource(Env* env) : env_(env) {}
  ~PubSubReadableResource() {}

  Status Init(const string& input, const std::vector<string>& metadata) {
    mutex_lock l(mu_);
    timeout_ = 10 * 1000;

    std::vector<string> parts = str_util::Split(input, "/");
    if (parts.size() != 4) {
      return errors::InvalidArgument("invalid input: ", input);
    }
    // projects/<project-id>/subscriptions/<subscription-id>
    auto subcription =
        pubsub::Subscription(std::move(parts[1]), std::move(parts[3]));

    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find("endpoint=") == 0) {
        parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid configuration: ",
                                         metadata[i]);
        }
        tensorflow::setenv("PUBSUB_EMULATOR_HOST", parts[1].c_str(),
                           /*overwrite=*/0);
      } else if (metadata[i].find("timeout=") == 0) {
        parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2 || !strings::safe_strtou64(parts[1], &timeout_)) {
          return errors::InvalidArgument("invalid configuration: ",
                                         metadata[i]);
        }
      }
    }

    connection_ = pubsub::MakeSubscriberConnection(subcription);
    return Status::OK();
  }
  Status Read(std::function<Status(const TensorShape& shape, Tensor** id_tensor,
                                   Tensor** data_tensor, Tensor** time_tensor)>
                  allocate_func) {
    int message_count = 0;
    std::vector<tstring> message_ids;
    std::vector<tstring> message_datas;
    std::vector<int64> message_times;
    Tensor* id_tensor;
    Tensor* data_tensor;
    Tensor* time_tensor;

    auto subscriber = pubsub::Subscriber(connection_);
    auto session = subscriber.Subscribe(
        [&](pubsub::Message const& m, pubsub::AckHandler h) {
          message_ids.emplace_back(std::move(m.message_id()));
          message_datas.emplace_back(std::move(m.data()));
          message_times.emplace_back(
              std::chrono::time_point_cast<std::chrono::milliseconds>(
                  m.publish_time())
                  .time_since_epoch()
                  .count());
          {
            mutex_lock lk(this->mu_);
            ++message_count;
          }
          std::move(h).ack();
        });
    {
      mutex_lock lk(mu_);
      auto condition = Condition(&HaveMessage, &message_count);
      if (timeout_ > 0) {
        mu_.AwaitWithDeadline(
            condition, env_->NowNanos() + timeout_ * EnvTime::kMillisToNanos);
      } else {
        mu_.Await(condition);
      }
    }
    session.cancel();
    auto status = session.get();
    if (!status.ok()) {
      return Status(static_cast<tensorflow::error::Code>(status.code()),
                    status.message());
    }
    TF_RETURN_IF_ERROR(allocate_func(TensorShape({message_count}), &id_tensor,
                                     &data_tensor, &time_tensor));
    if (message_count == 0) return Status::OK();

    std::move(message_ids.begin(), message_ids.end(),
              id_tensor->flat<tstring>().data());
    std::move(message_datas.begin(), message_datas.end(),
              data_tensor->flat<tstring>().data());
    std::move(message_times.begin(), message_times.end(),
              time_tensor->flat<int64>().data());
    return Status::OK();
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return "PubSubReadableResource";
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  uint64 timeout_ TF_GUARDED_BY(mu_);

  std::shared_ptr<pubsub::SubscriberConnection> connection_ = nullptr;

  static bool HaveMessage(int* message_count) { return *message_count > 0; }
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
    return Status::OK();
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
          return Status::OK();
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
