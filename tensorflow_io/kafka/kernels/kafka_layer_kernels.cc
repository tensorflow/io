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
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "rdkafkacpp.h"

namespace tensorflow {
namespace data {
namespace {

class LayerKafkaResource : public ResourceBase {
 public:
  LayerKafkaResource(Env* env) : env_(env) {}
  ~LayerKafkaResource() { Sync(); }

  Status Init(const string& topic, const int32 partition, const std::vector<string>& metadata) {
    std::unique_ptr<RdKafka::Conf> conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    std::unique_ptr<RdKafka::Conf> conf_topic(
        RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));

    string errstr;
    RdKafka::Conf::ConfResult result = RdKafka::Conf::CONF_UNKNOWN;

    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find("conf.topic.") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
            return errors::InvalidArgument("invalid topic configuration: ", metadata[i]);
        }
        result = conf_topic->set(parts[0].substr(11), parts[1], errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do topic configuration:", metadata[i], "error:", errstr);
        }
      } else if (metadata[i] != "" && metadata[i].find("conf.") == string::npos) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
            return errors::InvalidArgument("invalid global configuration: ", metadata[i]);
        }
        if ((result = conf->set(parts[0], parts[1], errstr)) != RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do global configuration: ", metadata[i], "error:", errstr);
        }
      }
      LOG(INFO) << "Kafka configuration: " << metadata[i];
    }
    if ((result = conf->set("default_topic_conf", conf_topic.get(), errstr)) != RdKafka::Conf::CONF_OK) {
      return errors::Internal("failed to set default_topic_conf:", errstr);
    }

    string bootstrap_servers;
    if ((result = conf->get("bootstrap.servers", bootstrap_servers)) != RdKafka::Conf::CONF_OK) {
      bootstrap_servers = "localhost:9092";
      if ((result = conf->set("bootstrap.servers", bootstrap_servers, errstr)) != RdKafka::Conf::CONF_OK) {
        return errors::Internal("failed to set bootstrap.servers [", bootstrap_servers, "]:", errstr);
      }
      LOG(INFO) << "Kafka default bootstrap server: " << bootstrap_servers;
    }

    producer_.reset(RdKafka::Producer::create(conf.get(), errstr));
    if (!(producer_.get() != nullptr)) {
      return errors::Internal("Failed to create producer:", errstr);
    }

    topic_.reset(RdKafka::Topic::create(producer_.get(), topic, conf_topic.get(), errstr));
    if (!(topic_.get() != nullptr)) {
      return errors::Internal("Failed to create topic ", topic, ":", errstr);
    }

    partition_ = partition;
    return Status::OK();
  }
  Status Write(const Tensor& content) {
    mutex_lock l(mu_);
    for (int64 i = 0; i < content.NumElements(); i++) {
      RdKafka::ErrorCode err = producer_->produce(
          topic_.get(), partition_, RdKafka::Producer::RK_MSG_COPY,
          (void*)content.flat<string>()(i).data(), content.flat<string>()(i).size(),
          NULL, NULL);
      if (!(err == RdKafka::ERR_NO_ERROR)) {
        return errors::Internal("Failed to produce message:", RdKafka::err2str(err));
      }
    }
    return Status::OK();
  }
  Status Sync() {
    if (producer_.get() != nullptr) {
      RdKafka::ErrorCode err = producer_->flush(timeout_);
      if (!(err == RdKafka::ERR_NO_ERROR)) {
        return errors::Internal("Failed to flush message:", RdKafka::err2str(err));
      }
    }
    return Status::OK();
  }
  string DebugString() const override {
    return "LayerKafkaResource";
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::Producer> producer_ GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::Topic> topic_ GUARDED_BY(mu_);
  int32 partition_ GUARDED_BY(mu_);
  static const int timeout_ = 5000;
};

class LayerKafkaInitOp : public ResourceOpKernel<LayerKafkaResource> {
 public:
  explicit LayerKafkaInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<LayerKafkaResource>(context) {
    env_ = context->env();
  }
 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<LayerKafkaResource>::Compute(context);

    const Tensor* topic_tensor;
    OP_REQUIRES_OK(context, context->input("topic", &topic_tensor));

    const Tensor* partition_tensor;
    OP_REQUIRES_OK(context, context->input("partition", &partition_tensor));

    const Tensor* metadata_tensor;
    OP_REQUIRES_OK(context, context->input("metadata", &metadata_tensor));
    std::vector<string> metadata;
    for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
      metadata.push_back(metadata_tensor->flat<string>()(i));
    }

    OP_REQUIRES_OK(context, resource_->Init(topic_tensor->scalar<string>()(), partition_tensor->scalar<int32>()(), metadata));
  }
  Status CreateResource(LayerKafkaResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new LayerKafkaResource(env_);
    return Status::OK();
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};


class LayerKafkaCallOp : public OpKernel {
 public:
  explicit LayerKafkaCallOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }

    const Tensor* content_tensor;
    OP_REQUIRES_OK(context, context->input("content", &content_tensor));

    LayerKafkaResource* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);

    OP_REQUIRES_OK(context, resource->Write(*content_tensor));
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class LayerKafkaSyncOp : public OpKernel {
 public:
  explicit LayerKafkaSyncOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    LayerKafkaResource* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);
    OP_REQUIRES_OK(context, resource->Sync());
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>LayerKafkaInit").Device(DEVICE_CPU),
                        LayerKafkaInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>LayerKafkaCall").Device(DEVICE_CPU),
                        LayerKafkaCallOp);
REGISTER_KERNEL_BUILDER(Name("IO>LayerKafkaSync").Device(DEVICE_CPU),
                        LayerKafkaSyncOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
