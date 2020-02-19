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

#include "rdkafkacpp.h"

namespace tensorflow {
namespace io {
namespace {
class KafkaEventCb : public RdKafka::EventCb {
 public:
  KafkaEventCb() : run_(true) {}

  bool run() { return run_; }

  void event_cb(RdKafka::Event& event) {
    switch (event.type()) {
      case RdKafka::Event::EVENT_ERROR:
        LOG(ERROR) << "EVENT_ERROR: "
                   << "(" << RdKafka::err2str(event.err())
                   << "): " << event.str();
        { run_ = (event.err() != RdKafka::ERR__ALL_BROKERS_DOWN); }
        break;
      case RdKafka::Event::EVENT_STATS:
        LOG(ERROR) << "EVENT_STATS: " << event.str();
        break;
      case RdKafka::Event::EVENT_LOG:
        LOG(ERROR) << "EVENT_LOG: " << event.severity() << "-"
                   << event.fac().c_str() << "-" << event.str().c_str();
        break;
      case RdKafka::Event::EVENT_THROTTLE:
        LOG(ERROR) << "EVENT_THROTTLE: " << event.throttle_time() << "ms by "
                   << event.broker_name() << " id " << (int)event.broker_id();
        break;
      default:
        LOG(ERROR) << "EVENT: " << event.type() << " ("
                   << RdKafka::err2str(event.err()) << "): " << event.str();
        break;
    }
  }

 private:
  mutable mutex mu_;
  bool run_ GUARDED_BY(mu_) = true;
};

class KafkaReadableResource : public ResourceBase {
 public:
  KafkaReadableResource(Env* env) : env_(env) {}
  virtual ~KafkaReadableResource() {
    if (consumer_.get()) {
      consumer_->unassign();
      consumer_->close();
      consumer_.reset(nullptr);
    }
  }

  virtual Status Init(const string& topic, const int32 partition,
                      const int64 offset, const std::vector<string>& metadata) {
    mutex_lock l(mu_);

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
          return errors::InvalidArgument("invalid topic configuration: ",
                                         metadata[i]);
        }
        result = conf_topic->set(parts[0].substr(11), parts[1], errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do topic configuration:",
                                  metadata[i], "error:", errstr);
        }
      } else if (metadata[i] != "" &&
                 metadata[i].find("conf.") == string::npos) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid topic configuration: ",
                                         metadata[i]);
        }
        if ((result = conf->set(parts[0], parts[1], errstr)) !=
            RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do global configuration: ",
                                  metadata[i], "error:", errstr);
        }
      }
      LOG(INFO) << "Kafka configuration: " << metadata[i];
    }
    if ((result = conf->set("default_topic_conf", conf_topic.get(), errstr)) !=
        RdKafka::Conf::CONF_OK) {
      return errors::Internal("failed to set default_topic_conf:", errstr);
    }

    // consumer.properties:
    //   bootstrap.servers=localhost:9092
    //   group.id=test-consumer-group
    string bootstrap_servers;
    if ((result = conf->get("bootstrap.servers", bootstrap_servers)) !=
        RdKafka::Conf::CONF_OK) {
      bootstrap_servers = "localhost:9092";
      if ((result = conf->set("bootstrap.servers", bootstrap_servers,
                              errstr)) != RdKafka::Conf::CONF_OK) {
        return errors::Internal("failed to set bootstrap.servers [",
                                bootstrap_servers, "]:", errstr);
      }
    }
    string group_id;
    if ((result = conf->get("group.id", group_id)) != RdKafka::Conf::CONF_OK) {
      group_id = "test-consumer-group";
      if ((result = conf->set("group.id", group_id, errstr)) !=
          RdKafka::Conf::CONF_OK) {
        return errors::Internal("failed to set group.id [", group_id,
                                "]:", errstr);
      }
    }

    // Always set enable.partition.eof=true
    if ((result = conf->set("enable.partition.eof", "true", errstr)) !=
        RdKafka::Conf::CONF_OK) {
      return errors::Internal("Failed to set enable.partition.eof=true :",
                              errstr);
    }

    if ((result = conf->set("event_cb", &kafka_event_cb_, errstr)) !=
        RdKafka::Conf::CONF_OK) {
      return errors::Internal("failed to set event_cb:", errstr);
    }

    consumer_.reset(RdKafka::KafkaConsumer::create(conf.get(), errstr));
    if (!consumer_.get()) {
      return errors::Internal("failed to create consumer:", errstr);
    }

    subscription_.reset(RdKafka::TopicPartition::create(topic, partition));
    std::vector<RdKafka::TopicPartition*> partitions;
    partitions.emplace_back(subscription_.get());

    subscription_->set_offset(offset);
    RdKafka::ErrorCode err = consumer_->assign(partitions);
    if (err != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to assign partition: ",
                              RdKafka::err2str(err));
    }

    return Status::OK();
  }
  Status Next(const int64 index,
              std::function<Status(const TensorShape& shape, Tensor** message,
                                   Tensor** key)>
                  allocate_func) {
    mutex_lock l(mu_);
    int64 total = 1024;
    std::vector<string> message_value, key_value;
    message_value.reserve(total);
    key_value.reserve(total);

    LOG(INFO) << "Kafka stream starts with current offset: "
              << subscription_->offset();
    std::unique_ptr<RdKafka::Message> message;
    int64 count = 0;
    while (consumer_.get() != nullptr && count < total) {
      if (!kafka_event_cb_.run()) {
        return errors::Internal("failed to consume due to all brokers down");
      }
      message.reset(consumer_->consume(timeout_));
      if (message->err() == RdKafka::ERR_NO_ERROR) {
        // Produce the line as output.
        message_value.emplace_back(string(
            static_cast<const char*>(message->payload()), message->len()));
        key_value.emplace_back(
            (message->key() != nullptr) ? string(*message->key()) : "");
        count++;
        continue;
      } else if (message->err() == RdKafka::ERR__TRANSPORT) {
        // Not return error here because consumer will try re-connect.
        LOG(ERROR) << "Broker transport failure: " << message->errstr();
      } else if (message->err() == RdKafka::ERR__PARTITION_EOF) {
        LOG(ERROR) << "EOF Message: " << message->errstr();
        consumer_.reset(nullptr);
        break;
      } else if (message->err() != RdKafka::ERR__TIMED_OUT) {
        LOG(ERROR) << "Failed to consume: " << message->errstr();
        return errors::Internal("Failed to consume: ", message->errstr());
      }
    }
    TensorShape shape({static_cast<int64>(message_value.size())});
    Tensor* message_tensor;
    Tensor* key_tensor;
    TF_RETURN_IF_ERROR(allocate_func(shape, &message_tensor, &key_tensor));
    for (size_t i = 0; i < message_value.size(); i++) {
      message_tensor->flat<string>()(i) = message_value[i];
      key_tensor->flat<string>()(i) = key_value[i];
    }
    return Status::OK();
  }
  Status Read(const int64 start, const int64 stop,
              std::function<Status(const TensorShape& shape, Tensor** message,
                                   Tensor** key)>
                  allocate_func) {
    mutex_lock l(mu_);

    int64 stop_offset;
    if (stop >= 0) {
      stop_offset = stop;
    } else if (stop == RdKafka::Topic::OFFSET_END) {
      stop_offset = RdKafka::Consumer::OffsetTail(0);
    } else if (stop <= RdKafka::Consumer::OffsetTail(0)) {
      stop_offset = stop;
    } else {
      return errors::InvalidArgument("stop offset ", stop, " not supported");
    }
    if (stop_offset <= RdKafka::Consumer::OffsetTail(0)) {
      int64 tail_offset = 0;
      TF_RETURN_IF_ERROR(Tail(&tail_offset));

      stop_offset =
          tail_offset + stop_offset - RdKafka::Consumer::OffsetTail(0);
    }

    std::vector<string> message_value, key_value;

    subscription_->set_offset(start);
    RdKafka::ErrorCode err = consumer_->seek((*subscription_), timeout_);
    if (err != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to seek partition: ",
                              RdKafka::err2str(err));
    }
    LOG(INFO) << "Kafka stream starts with current offset: "
              << subscription_->offset();
    int64 index = start;
    std::unique_ptr<RdKafka::Message> message;
    while (consumer_.get() != nullptr && index + 1 < stop_offset) {
      if (!kafka_event_cb_.run()) {
        return errors::Internal("failed to consume due to all brokers down");
      }
      message.reset(consumer_->consume(timeout_));
      if (message->err() == RdKafka::ERR_NO_ERROR) {
        // Produce the line as output.
        message_value.emplace_back(string(
            static_cast<const char*>(message->payload()), message->len()));
        key_value.emplace_back(
            (message->key() != nullptr) ? string(*message->key()) : "");
        index = message->offset();
        continue;
      } else if (message->err() == RdKafka::ERR__PARTITION_EOF) {
        LOG(ERROR) << "EOF Message: " << message->errstr();
        break;
      } else if (message->err() == RdKafka::ERR__TRANSPORT) {
        // Not return error here because consumer will try re-connect.
        LOG(ERROR) << "Broker transport failure: " << message->errstr();
      } else if (message->err() != RdKafka::ERR__TIMED_OUT) {
        LOG(ERROR) << "Failed to consume: " << message->errstr();
        return errors::Internal("Failed to consume: ", message->errstr());
      }
    }
    TensorShape shape({static_cast<int64>(message_value.size())});
    Tensor* message_tensor;
    Tensor* key_tensor;
    TF_RETURN_IF_ERROR(allocate_func(shape, &message_tensor, &key_tensor));
    for (size_t i = 0; i < message_value.size(); i++) {
      message_tensor->flat<string>()(i) = message_value[i];
      key_tensor->flat<string>()(i) = key_value[i];
    }
    return Status::OK();
  }
  Status Spec(const int64 start, const int64 stop, int64* start_offset,
              int64* stop_offset) {
    mutex_lock l(mu_);

    if (start >= 0) {
      *start_offset = start;
    } else if (start == RdKafka::Topic::OFFSET_END) {
      *start_offset = RdKafka::Consumer::OffsetTail(0);
    } else if (start <= RdKafka::Consumer::OffsetTail(0)) {
      *start_offset = start;
    } else {
      return errors::InvalidArgument("start offset ", start, " not supported");
    }

    if (stop >= 0) {
      *stop_offset = stop;
    } else if (stop == RdKafka::Topic::OFFSET_END) {
      *stop_offset = RdKafka::Consumer::OffsetTail(0);
    } else if (stop <= RdKafka::Consumer::OffsetTail(0)) {
      *stop_offset = stop;
    } else {
      return errors::InvalidArgument("stop offset ", stop, " not supported");
    }

    if (*start_offset <= RdKafka::Consumer::OffsetTail(0) ||
        *stop_offset <= RdKafka::Consumer::OffsetTail(0)) {
      int64 tail_offset = 0;
      TF_RETURN_IF_ERROR(Tail(&tail_offset));

      if (*start_offset <= RdKafka::Consumer::OffsetTail(0)) {
        *start_offset =
            tail_offset + *start_offset - RdKafka::Consumer::OffsetTail(0);
      }
      if (*stop_offset <= RdKafka::Consumer::OffsetTail(0)) {
        *stop_offset =
            tail_offset + *stop_offset - RdKafka::Consumer::OffsetTail(0);
      }
    }

    return Status::OK();
  }
  string DebugString() const override { return "KafkaBaseResource"; }

 protected:
  Status Tail(int64* tail_offset) {
    // Resolve tail message
    int64 saved = subscription_->offset();

    subscription_->set_offset(RdKafka::Consumer::OffsetTail(1));
    RdKafka::ErrorCode err = consumer_->seek(*subscription_, timeout_);
    if (err != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to seek tail -1: ",
                              RdKafka::err2str(err));
    }
    std::unique_ptr<RdKafka::Message> message;
    do {
      message.reset(consumer_->consume(timeout_));
    } while (message->err() == RdKafka::ERR__TRANSPORT);
    if (message->err() != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to consume tail message: ",
                              RdKafka::err2str(message->err()));
    }
    *tail_offset = message->offset() + 1;
    LOG(INFO) << "Kafka tail: " << *tail_offset;

    subscription_->set_offset(saved);
    err = consumer_->seek(*subscription_, timeout_);
    if (err != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to seek back saved: ",
                              RdKafka::err2str(err));
    }

    return Status::OK();
  }
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::TopicPartition> subscription_ GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::KafkaConsumer> consumer_ GUARDED_BY(mu_);
  KafkaEventCb kafka_event_cb_ = KafkaEventCb();
  static const int timeout_ = 5000;
};

class KafkaReadableInitOp : public ResourceOpKernel<KafkaReadableResource> {
 public:
  explicit KafkaReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<KafkaReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<KafkaReadableResource>::Compute(context);

    const Tensor* topic_tensor;
    OP_REQUIRES_OK(context, context->input("topic", &topic_tensor));
    const string& topic = topic_tensor->scalar<string>()();

    const Tensor* partition_tensor;
    OP_REQUIRES_OK(context, context->input("partition", &partition_tensor));
    const int32 partition = partition_tensor->scalar<int32>()();

    const Tensor* offset_tensor;
    OP_REQUIRES_OK(context, context->input("offset", &offset_tensor));
    const int64 offset = offset_tensor->scalar<int64>()();

    const Tensor* metadata_tensor;
    OP_REQUIRES_OK(context, context->input("metadata", &metadata_tensor));
    std::vector<string> metadata;
    for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
      metadata.push_back(metadata_tensor->flat<string>()(i));
    }

    OP_REQUIRES_OK(context,
                   resource_->Init(topic, partition, offset, metadata));
  }
  Status CreateResource(KafkaReadableResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new KafkaReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class KafkaReadableNextOp : public OpKernel {
 public:
  explicit KafkaReadableNextOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    KafkaReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* index_tensor;
    OP_REQUIRES_OK(context, context->input("index", &index_tensor));
    const int64 index = index_tensor->scalar<int64>()();

    OP_REQUIRES_OK(
        context,
        resource->Next(
            index,
            [&](const TensorShape& shape, Tensor** message,
                Tensor** key) -> Status {
              TF_RETURN_IF_ERROR(context->allocate_output(0, shape, message));
              TF_RETURN_IF_ERROR(context->allocate_output(1, shape, key));
              return Status::OK();
            }));
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class KafkaReadableReadOp : public OpKernel {
 public:
  explicit KafkaReadableReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    KafkaReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    const int64 start = start_tensor->scalar<int64>()();

    const Tensor* stop_tensor;
    OP_REQUIRES_OK(context, context->input("stop", &stop_tensor));
    const int64 stop = stop_tensor->scalar<int64>()();

    OP_REQUIRES_OK(
        context,
        resource->Read(
            start, stop,
            [&](const TensorShape& shape, Tensor** message,
                Tensor** key) -> Status {
              TF_RETURN_IF_ERROR(context->allocate_output(0, shape, message));
              TF_RETURN_IF_ERROR(context->allocate_output(1, shape, key));
              return Status::OK();
            }));
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class KafkaReadableSpecOp : public OpKernel {
 public:
  explicit KafkaReadableSpecOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    KafkaReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    const int64 start = start_tensor->scalar<int64>()();

    const Tensor* stop_tensor;
    OP_REQUIRES_OK(context, context->input("stop", &stop_tensor));
    const int64 stop = stop_tensor->scalar<int64>()();

    int64 start_offset, stop_offset;
    OP_REQUIRES_OK(context,
                   resource->Spec(start, stop, &start_offset, &stop_offset));

    Tensor* start_offset_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                     &start_offset_tensor));
    start_offset_tensor->scalar<int64>()() = start_offset;

    Tensor* stop_offset_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                     &stop_offset_tensor));
    stop_offset_tensor->scalar<int64>()() = stop_offset;
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
/*
class KafkaIterableInitOp : public ResourceOpKernel<KafkaIterableResource> {
 public:
  explicit KafkaIterableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<KafkaIterableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<KafkaIterableResource>::Compute(context);

    const Tensor* topic_tensor;
    OP_REQUIRES_OK(context, context->input("topic", &topic_tensor));
    const string& topic = topic_tensor->scalar<string>()();

    const Tensor* partition_tensor;
    OP_REQUIRES_OK(context, context->input("partition", &partition_tensor));
    const int32 partition = partition_tensor->scalar<int32>()();

    const Tensor* offset_tensor;
    OP_REQUIRES_OK(context, context->input("offset", &offset_tensor));
    const int64 offset = offset_tensor->scalar<int64>()();

    const Tensor* metadata_tensor;
    OP_REQUIRES_OK(context, context->input("metadata", &metadata_tensor));
    std::vector<string> metadata;
    for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
      metadata.push_back(metadata_tensor->flat<string>()(i));
    }

    OP_REQUIRES_OK(context,
                   resource_->Init(topic, partition, offset, metadata));
  }
  Status CreateResource(KafkaIterableResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new KafkaIterableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
*/
class LayerKafkaResource : public ResourceBase {
 public:
  LayerKafkaResource(Env* env) : env_(env) {}
  ~LayerKafkaResource() { Sync(); }

  Status Init(const string& topic, const int32 partition,
              const std::vector<string>& metadata) {
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
          return errors::InvalidArgument("invalid topic configuration: ",
                                         metadata[i]);
        }
        result = conf_topic->set(parts[0].substr(11), parts[1], errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do topic configuration:",
                                  metadata[i], "error:", errstr);
        }
      } else if (metadata[i] != "" &&
                 metadata[i].find("conf.") == string::npos) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid global configuration: ",
                                         metadata[i]);
        }
        if ((result = conf->set(parts[0], parts[1], errstr)) !=
            RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do global configuration: ",
                                  metadata[i], "error:", errstr);
        }
      }
      LOG(INFO) << "Kafka configuration: " << metadata[i];
    }
    if ((result = conf->set("default_topic_conf", conf_topic.get(), errstr)) !=
        RdKafka::Conf::CONF_OK) {
      return errors::Internal("failed to set default_topic_conf:", errstr);
    }

    string bootstrap_servers;
    if ((result = conf->get("bootstrap.servers", bootstrap_servers)) !=
        RdKafka::Conf::CONF_OK) {
      bootstrap_servers = "localhost:9092";
      if ((result = conf->set("bootstrap.servers", bootstrap_servers,
                              errstr)) != RdKafka::Conf::CONF_OK) {
        return errors::Internal("failed to set bootstrap.servers [",
                                bootstrap_servers, "]:", errstr);
      }
      LOG(INFO) << "Kafka default bootstrap server: " << bootstrap_servers;
    }

    producer_.reset(RdKafka::Producer::create(conf.get(), errstr));
    if (!(producer_.get() != nullptr)) {
      return errors::Internal("Failed to create producer:", errstr);
    }

    topic_.reset(RdKafka::Topic::create(producer_.get(), topic,
                                        conf_topic.get(), errstr));
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
          (void*)content.flat<string>()(i).data(),
          content.flat<string>()(i).size(), NULL, NULL);
      if (!(err == RdKafka::ERR_NO_ERROR)) {
        return errors::Internal("Failed to produce message:",
                                RdKafka::err2str(err));
      }
    }
    return Status::OK();
  }
  Status Sync() {
    if (producer_.get() != nullptr) {
      RdKafka::ErrorCode err = producer_->flush(timeout_);
      if (!(err == RdKafka::ERR_NO_ERROR)) {
        return errors::Internal("Failed to flush message:",
                                RdKafka::err2str(err));
      }
    }
    return Status::OK();
  }
  string DebugString() const override { return "LayerKafkaResource"; }

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

    OP_REQUIRES_OK(context, resource_->Init(topic_tensor->scalar<string>()(),
                                            partition_tensor->scalar<int32>()(),
                                            metadata));
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
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
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
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);
    OP_REQUIRES_OK(context, resource->Sync());
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>KafkaReadableInit").Device(DEVICE_CPU),
                        KafkaReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>KafkaReadableNext").Device(DEVICE_CPU),
                        KafkaReadableNextOp);
REGISTER_KERNEL_BUILDER(Name("IO>KafkaReadableRead").Device(DEVICE_CPU),
                        KafkaReadableReadOp);
REGISTER_KERNEL_BUILDER(Name("IO>KafkaReadableSpec").Device(DEVICE_CPU),
                        KafkaReadableSpecOp);
REGISTER_KERNEL_BUILDER(Name("IO>LayerKafkaInit").Device(DEVICE_CPU),
                        LayerKafkaInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>LayerKafkaCall").Device(DEVICE_CPU),
                        LayerKafkaCallOp);
REGISTER_KERNEL_BUILDER(Name("IO>LayerKafkaSync").Device(DEVICE_CPU),
                        LayerKafkaSyncOp);
}  // namespace
}  // namespace io
}  // namespace tensorflow
