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

#include "pulsar/Client.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {
namespace io {
namespace {

class PulsarResourceBase : public ResourceBase {
 public:
  PulsarResourceBase() = default;

  virtual ~PulsarResourceBase() {
    if (client_.get()) {
      client_->close();
      client_.reset(nullptr);
    }
  }

 protected:
  mutable mutex mu_;
  std::unique_ptr<pulsar::Client> client_ TF_GUARDED_BY(mu_);

  void Init(const std::string& service_url) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    client_.reset(new pulsar::Client(service_url));
  }
};

class PulsarReadableResource final : public PulsarResourceBase {
 public:
  Status Init(const std::string& service_url, const std::string& topic,
              const std::string& subscription, int64 ack_grouping_time) {
    mutex_lock l(mu_);
    PulsarResourceBase::Init(service_url);

    pulsar::ConsumerConfiguration conf;
    conf.setConsumerType(pulsar::ConsumerFailover);
    conf.setSubscriptionInitialPosition(pulsar::InitialPositionEarliest);
    conf.setAckGroupingTimeMs(ack_grouping_time);

    auto result = client_->subscribe(topic, subscription, conf, consumer_);
    if (result != pulsar::ResultOk) {
      return errors::Internal("failed to subscribe ", topic,
                              " subscription: ", subscription,
                              " error: ", pulsar::strResult(result));
    }

    LOG(INFO) << "Subscribing to the pulsar topic: " << topic
              << " with subscription: " << subscription;
    return Status::OK();
  }

  Status Next(const int32 timeout, const int32 poll_timeout,
              std::function<Status(const TensorShape& shape, Tensor** message,
                                   Tensor** key, Tensor** continue_fetch)>
                  allocate_func) {
    mutex_lock l(mu_);

    constexpr size_t max_num_messages = 1024;
    std::vector<std::string> values;
    std::vector<std::string> keys;
    values.reserve(max_num_messages);
    keys.reserve(max_num_messages);

    int32 elapsed_time = 0;
    int num_messages = 0;
    while (elapsed_time < timeout && num_messages < max_num_messages) {
      pulsar::Message message;
      auto result = consumer_.receive(message, poll_timeout);
      if (result == pulsar::ResultOk) {
        keys.emplace_back(message.hasPartitionKey() ? message.getPartitionKey()
                                                    : "");
        values.emplace_back(message.getDataAsString());
        num_messages++;
        elapsed_time = 0;  // reset the current timeout
        consumer_.acknowledgeAsync(message, [&](pulsar::Result result) {
          if (result != pulsar::ResultOk) {
            LOG(ERROR) << "Failed to acknowledge " << message.getMessageId();
          }
        });
      } else if (result == pulsar::ResultTimeout) {
        elapsed_time += poll_timeout;
      } else {
        return errors::Internal("failed to receive messages, error: ",
                                pulsar::strResult(result));
      }
    }

    TensorShape shape({static_cast<int32>(values.size())});
    Tensor* value_tensor;
    Tensor* key_tensor;
    Tensor* continue_fetch_tensor;
    TF_RETURN_IF_ERROR(allocate_func(shape, &value_tensor, &key_tensor,
                                     &continue_fetch_tensor));

    // If no messages were received when timeout exceeded, we treat it as a
    // failure and don't continue receiving messages.
    continue_fetch_tensor->scalar<int64>()() = (values.empty() ? 0 : 1);
    for (size_t i = 0; i < values.size(); i++) {
      value_tensor->flat<tstring>()(i) = values[i];
      key_tensor->flat<tstring>()(i) = keys[i];
    }

    return Status::OK();
  }

  std::string DebugString() const override { return "PulsarReadableResource"; }

 private:
  pulsar::Consumer consumer_;
};

class PulsarReadableInitOp : public ResourceOpKernel<PulsarReadableResource> {
 public:
  explicit PulsarReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<PulsarReadableResource>(context) {}

 private:
  void Compute(OpKernelContext* context) override {
    mutex_lock l(mu_);
    ResourceOpKernel<PulsarReadableResource>::Compute(context);

    const Tensor* service_url_tensor;
    OP_REQUIRES_OK(context, context->input("service_url", &service_url_tensor));
    const std::string service_url = service_url_tensor->flat<tstring>()(0);

    const Tensor* topic_tensor;
    OP_REQUIRES_OK(context, context->input("topic", &topic_tensor));
    const std::string topic = topic_tensor->flat<tstring>()(0);

    const Tensor* subscription_tensor;
    OP_REQUIRES_OK(context,
                   context->input("subscription", &subscription_tensor));
    const std::string subscription = subscription_tensor->flat<tstring>()(0);

    const Tensor* ack_grouping_time_tensor;
    OP_REQUIRES_OK(context, context->input("ack_grouping_time",
                                           &ack_grouping_time_tensor));
    const int64 ack_grouping_time = ack_grouping_time_tensor->scalar<int64>()();

    OP_REQUIRES_OK(context, resource_->Init(service_url, topic, subscription,
                                            ack_grouping_time));
  }

  Status CreateResource(PulsarReadableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new PulsarReadableResource();
    return Status::OK();
  }

 private:
  mutable mutex mu_;
};

class PulsarReadableNextOp : public OpKernel {
 public:
  explicit PulsarReadableNextOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    PulsarReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* timeout_tensor;
    OP_REQUIRES_OK(context, context->input("timeout", &timeout_tensor));
    const int64 timeout = timeout_tensor->scalar<int64>()();

    const Tensor* poll_timeout_tensor;
    OP_REQUIRES_OK(context,
                   context->input("poll_timeout", &poll_timeout_tensor));
    const int64 poll_timeout = poll_timeout_tensor->scalar<int64>()();

    OP_REQUIRES_OK(
        context,
        resource->Next(
            timeout, poll_timeout,
            [&](const TensorShape& shape, Tensor** message, Tensor** key,
                Tensor** continue_fetch) -> Status {
              TF_RETURN_IF_ERROR(context->allocate_output(0, shape, message));
              TF_RETURN_IF_ERROR(context->allocate_output(1, shape, key));
              TF_RETURN_IF_ERROR(
                  context->allocate_output(2, TensorShape({}), continue_fetch));
              return Status::OK();
            }));
  }

 private:
  mutable mutex mu_;
};

class PulsarWritableResource final : public PulsarResourceBase {
 public:
  Status Init(const std::string& service_url, const std::string& topic) {
    mutex_lock l(mu_);
    PulsarResourceBase::Init(service_url);
    index_ = 0;

    pulsar::ProducerConfiguration conf;
    conf.setPartitionsRoutingMode(
        pulsar::ProducerConfiguration::RoundRobinDistribution);

    auto result = client_->createProducer(topic, conf, producer_);
    if (result != pulsar::ResultOk) {
      return errors::Internal("failed to create producer for topic: ", topic,
                              " error: ", pulsar::strResult(result));
    }

    LOG(INFO) << "Created producer on pulsar topic: " << topic;
    return Status::OK();
  }

  Status WriteAsync(const std::string& value, const std::string& key) {
    mutex_lock l(mu_);
    pulsar::MessageBuilder builder;
    if (!key.empty()) {
      builder.setPartitionKey(key);
    }
    auto send_async_result = pulsar::ResultOk;
    producer_.sendAsync(
        builder.setContent(value).build(),
        [index = index_, &send_async_result](pulsar::Result result,
                                             const pulsar::MessageId& id) {
          if (result != pulsar::ResultOk) {
            LOG(ERROR) << "failed to send message-" << index << ": " << result;
          }
          send_async_result = result;
        });
    // sendAsync may fail immediately caused by queue is full
    if (send_async_result != pulsar::ResultOk) {
      return errors::Internal("sendAsync failed for index: ", index_,
                              " error: ", pulsar::strResult(send_async_result));
    }
    index_++;
    return Status::OK();
  }

  Status Flush() {
    mutex_lock l(mu_);
    auto result = producer_.flush();
    if (result != pulsar::ResultOk) {
      return errors::Internal("failed to flush: ", pulsar::strResult(result));
    }
    return Status::OK();
  }

  std::string DebugString() const override { return "PulsarWritableResource"; }

 private:
  pulsar::Producer producer_;
  unsigned long index_;
};

class PulsarWritableInitOp : public ResourceOpKernel<PulsarWritableResource> {
 public:
  explicit PulsarWritableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<PulsarWritableResource>(context) {}

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<PulsarWritableResource>::Compute(context);

    const Tensor* service_url_tensor;
    OP_REQUIRES_OK(context, context->input("service_url", &service_url_tensor));
    const std::string service_url = service_url_tensor->flat<tstring>()(0);

    const Tensor* topic_tensor;
    OP_REQUIRES_OK(context, context->input("topic", &topic_tensor));
    const std::string topic = topic_tensor->flat<tstring>()(0);

    OP_REQUIRES_OK(context, resource_->Init(service_url, topic));
  }

  Status CreateResource(PulsarWritableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new PulsarWritableResource();
    return Status::OK();
  }

 private:
  mutable mutex mu_;
};

class PulsarWritableWriteOp : public OpKernel {
 public:
  explicit PulsarWritableWriteOp(OpKernelConstruction* context)
      : OpKernel(context) {}

 private:
  void Compute(OpKernelContext* context) override {
    PulsarWritableResource* resource;

    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* value_tensor;
    OP_REQUIRES_OK(context, context->input("value", &value_tensor));
    const std::string value = value_tensor->flat<tstring>()(0);

    const Tensor* key_tensor;
    OP_REQUIRES_OK(context, context->input("key", &key_tensor));
    const std::string key = key_tensor->flat<tstring>()(0);

    OP_REQUIRES_OK(context, resource->WriteAsync(value, key));
  }
};

class PulsarWritableFlushOp : public OpKernel {
 public:
  explicit PulsarWritableFlushOp(OpKernelConstruction* context)
      : OpKernel(context) {}

 private:
  void Compute(OpKernelContext* context) override {
    PulsarWritableResource* resource;

    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    OP_REQUIRES_OK(context, resource->Flush());
  }
};

REGISTER_KERNEL_BUILDER(Name("IO>PulsarReadableInit").Device(DEVICE_CPU),
                        PulsarReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>PulsarReadableNext").Device(DEVICE_CPU),
                        PulsarReadableNextOp);
REGISTER_KERNEL_BUILDER(Name("IO>PulsarWritableInit").Device(DEVICE_CPU),
                        PulsarWritableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>PulsarWritableWrite").Device(DEVICE_CPU),
                        PulsarWritableWriteOp);
REGISTER_KERNEL_BUILDER(Name("IO>PulsarWritableFlush").Device(DEVICE_CPU),
                        PulsarWritableFlushOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
