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

#include "tensorflow_io/core/kernels/io_interface.h"

#include "rdkafkacpp.h"

#include <unordered_map>

namespace tensorflow {
namespace data {

class KafkaEventCb : public RdKafka::EventCb {
public:
  KafkaEventCb()
  : run_(true) {}

  bool run() {
    return run_;
  }

  void event_cb (RdKafka::Event &event) {
    switch (event.type()) {
    case RdKafka::Event::EVENT_ERROR:
      LOG(ERROR) << "EVENT_ERROR: " << "(" << RdKafka::err2str(event.err()) << "): " << event.str();
      {
        run_ = (event.err() != RdKafka::ERR__ALL_BROKERS_DOWN);
      }
      break;
    case RdKafka::Event::EVENT_STATS:
      LOG(ERROR) << "EVENT_STATS: " << event.str();
      break;
    case RdKafka::Event::EVENT_LOG:
      LOG(ERROR) << "EVENT_LOG: " << event.severity() << "-" << event.fac().c_str() << "-" << event.str().c_str();
      break;
    case RdKafka::Event::EVENT_THROTTLE:
      LOG(ERROR) << "EVENT_THROTTLE: " << event.throttle_time() << "ms by " << event.broker_name() << " id " << (int)event.broker_id();
      break;
    default:
      LOG(ERROR) << "EVENT: " << event.type() << " (" << RdKafka::err2str(event.err()) << "): " << event.str();
      break;
    }
  }
private:
  mutable mutex mu_;
  bool run_ GUARDED_BY(mu_) = true;
};

class KafkaIterable : public IOIterableInterface {
 public:
  KafkaIterable(Env* env)
  : env_(env) {}

  ~KafkaIterable() {}
  Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) override {
    std::unique_ptr<RdKafka::Conf> conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    std::unique_ptr<RdKafka::Conf> conf_topic(
        RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));

    string errstr;
    RdKafka::Conf::ConfResult result = RdKafka::Conf::CONF_UNKNOWN;

    eof_ = true;
    timeout_ = 1000;
    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find_first_of("conf.eof") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid timeout configuration: ", metadata[i]);
        }
        eof_ = (parts[1] != "0");
      } else if (metadata[i].find_first_of("conf.timeout") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2 || !strings::safe_strto64(parts[1], &timeout_)) {
          return errors::InvalidArgument("invalid timeout configuration: ", metadata[i]);
        }
      } else if (metadata[i].find_first_of("conf.topic.") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
            return errors::InvalidArgument("invalid topic configuration: ", metadata[i]);
        }
        result = conf_topic->set(parts[0].substr(11), parts[1], errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do topic configuration:", metadata[i], "error:", errstr);
        }
      } else if (metadata[i] != "" && metadata[i].find_first_of("conf.") == string::npos) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
            return errors::InvalidArgument("invalid topic configuration: ", metadata[i]);
        }
        if ((result = conf->set(parts[0], parts[1], errstr)) != RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do global configuration: ", metadata[i], "error:", errstr);
        }
      }
    }
    if ((result = conf->set("default_topic_conf", conf_topic.get(), errstr)) != RdKafka::Conf::CONF_OK) {
      return errors::Internal("failed to set default_topic_conf:", errstr);
    }

    // consumer.properties:
    //   bootstrap.servers=localhost:9092
    //   group.id=test-consumer-group
    string bootstrap_servers;
    if ((result = conf->get("bootstrap.servers", bootstrap_servers)) != RdKafka::Conf::CONF_OK) {
      bootstrap_servers = "localhost:9092";
      if ((result = conf->set("bootstrap.servers", bootstrap_servers, errstr)) != RdKafka::Conf::CONF_OK) {
        return errors::Internal("failed to set bootstrap.servers [", bootstrap_servers, "]:", errstr);
      }
    }
    string group_id;
    if ((result = conf->get("group.id", group_id)) != RdKafka::Conf::CONF_OK) {
      group_id = "test-consumer-group";
      if ((result = conf->set("group.id", group_id, errstr)) != RdKafka::Conf::CONF_OK) {
        return errors::Internal("failed to set group.id [", group_id, "]:", errstr);
      }
    }
    if ((result = conf->set("event_cb", &kafka_event_cb_, errstr)) != RdKafka::Conf::CONF_OK) {
      return errors::Internal("failed to set event_cb:", errstr);
    }

    // TODO: multiple topic and partitions

    const string& entry = input[0];
    std::vector<string> parts = str_util::Split(entry, ":");
    string topic = parts[0];
    int32 partition = 0;
    if (parts.size() > 1) {
      if (!strings::safe_strto32(parts[1], &partition)) {
        return errors::InvalidArgument("invalid parameters: ", entry);
      }
    }

    int64 start = 0;
    if (parts.size() > 2) {
      if (!strings::safe_strto64(parts[2], &start)) {
        return errors::InvalidArgument("invalid parameters: ", entry);
      }
    }
    subscription_.reset(RdKafka::TopicPartition::create(topic, partition, start));
    start = subscription_->offset();

    offset_ = start;

    int64 stop = -1;
    if (parts.size() > 3) {
      if (!strings::safe_strto64(parts[3], &stop)) {
        return errors::InvalidArgument("invalid parameters: ", entry);
      }
    }
    range_ = std::pair<int64, int64>(start, stop);

    consumer_.reset(RdKafka::KafkaConsumer::create(conf.get(), errstr));
    if (!consumer_.get()) {
      return errors::Internal("failed to create consumer:", errstr);
    }

    std::vector<RdKafka::TopicPartition*> partitions;
    partitions.emplace_back(subscription_.get());
    RdKafka::ErrorCode err = consumer_->assign(partitions);
    if (err != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to assign partition: ", RdKafka::err2str(err));
    }

    return Status::OK();
  }
  Status Next(const int64 capacity, const Tensor& component, Tensor* tensor, int64* record_read) override {
    *record_read = 0;
    while (consumer_.get() != nullptr && (*record_read) < capacity) {
      if (!kafka_event_cb_.run()) {
        return errors::Internal("failed to consume due to all brokers down");
      }
      if (range_.second >= 0 && (subscription_->offset() >= range_.second || offset_ >= range_.second)) {
        // EOF of topic
        consumer_.reset(nullptr);
        return Status::OK();
      }

      std::unique_ptr<RdKafka::Message> message(consumer_->consume(timeout_));
      if (message->err() == RdKafka::ERR_NO_ERROR) {
         // Produce the line as output.
         tensor->flat<string>()((*record_read)) = std::string(static_cast<const char*>(message->payload()), message->len());
         // Sync offset
         offset_ = message->offset();
         (*record_read)++;
         continue;
      }
      if (message->err() == RdKafka::ERR__PARTITION_EOF) {
        LOG(INFO) << "Partition reach EOF, current offset: " << offset_;
        if (eof_) {
          consumer_.reset(nullptr);
          return Status::OK();
        }
      }
      else if (message->err() == RdKafka::ERR__TRANSPORT) {
        // Not return error here because consumer will try re-connect.
        LOG(ERROR) << "Broker transport failure: " << message->errstr();
      }
      else if (message->err() != RdKafka::ERR__TIMED_OUT) {
        LOG(ERROR) << "Failed to consume: " << message->errstr();
        return errors::Internal("Failed to consume: ", message->errstr());
      }
    }
    return Status::OK();
  }
  Status Spec(const Tensor& component, PartialTensorShape* shape, DataType* dtype) override {
    *shape = PartialTensorShape({-1});
    *dtype = DT_STRING;
    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("KafkaIterable[]");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::pair<string, int32> range_ GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::TopicPartition> subscription_ GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::KafkaConsumer> consumer_ GUARDED_BY(mu_);
  KafkaEventCb kafka_event_cb_ = KafkaEventCb();
  int64 timeout_ GUARDED_BY(mu_);
  bool eof_ GUARDED_BY(mu_);
  int64 offset_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("KafkaIterableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<KafkaIterable>);
REGISTER_KERNEL_BUILDER(Name("KafkaIterableNext").Device(DEVICE_CPU),
                        IOIterableNextOp<KafkaIterable>);
REGISTER_KERNEL_BUILDER(Name("KafkaIndexableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<IOIndexableImplementation<KafkaIterable>>);
REGISTER_KERNEL_BUILDER(Name("KafkaIndexableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<IOIndexableImplementation<KafkaIterable>>);
REGISTER_KERNEL_BUILDER(Name("KafkaIndexableGetItem").Device(DEVICE_CPU),
                        IOIndexableGetItemOp<IOIndexableImplementation<KafkaIterable>>);

}  // namespace data
}  // namespace tensorflow
