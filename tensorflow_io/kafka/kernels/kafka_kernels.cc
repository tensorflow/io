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
#include "api/DataFile.hh"
#include "api/Compiler.hh"
#include "api/Generic.hh"
#include "api/Stream.hh"
#include "api/Validator.hh"

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

class KafkaReadable : public IOReadableInterface {
 public:
  KafkaReadable(Env* env) : env_(env) {}

  ~KafkaReadable() {}
  Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) override {
    std::unique_ptr<RdKafka::Conf> conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    std::unique_ptr<RdKafka::Conf> conf_topic(
        RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));

    string errstr;
    RdKafka::Conf::ConfResult result = RdKafka::Conf::CONF_UNKNOWN;

    eof_ = true;
    timeout_ = 2000;
    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find_first_of("conf.eof") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid bounded configuration: ", metadata[i]);
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

    consumer_.reset(RdKafka::KafkaConsumer::create(conf.get(), errstr));
    if (!consumer_.get()) {
      return errors::Internal("failed to create consumer:", errstr);
    }

    // TODO: multiple topic and partitions
    const string& entry = input[0];
    std::vector<string> parts = str_util::Split(entry, ":");
    if (parts.size() != 3 && parts.size() != 4) {
      return errors::InvalidArgument("invalid input: ", entry);
    }
    string topic = parts[0];
    int32 partition = 0;
    if (!strings::safe_strto32(parts[1], &partition)) {
      return errors::InvalidArgument("invalid parameters: ", entry);
    }
    subscription_.reset(RdKafka::TopicPartition::create(topic, partition));
    std::vector<RdKafka::TopicPartition*> partitions;
    partitions.emplace_back(subscription_.get());

    offset_head_ = 0;
    offset_tail_ = -1;
    if (!strings::safe_strto64(parts[2], &offset_head_)) {
      return errors::InvalidArgument("invalid parameters: ", entry);
    }
    subscription_->set_offset(offset_head_);
    RdKafka::ErrorCode err = consumer_->assign(partitions);
    if (err != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to assign partition: ", RdKafka::err2str(err));
    }
    // If offset tail is specified then resolve
    if (parts.size() == 4) {
      std::unique_ptr<RdKafka::Message> message;
      do {
        message.reset(consumer_->consume(timeout_));
      } while (message->err() == RdKafka::ERR__TRANSPORT);
      if (message->err() != RdKafka::ERR_NO_ERROR) {
        return errors::Internal("failed to consume head message: ", RdKafka::err2str(message->err()));
      }
      offset_head_ = message->offset();

      if (!strings::safe_strto64(parts[3], &offset_tail_)) {
        return errors::InvalidArgument("invalid parameters: ", entry);
      }
      if (offset_tail_ == -1) {
        subscription_->set_offset(RdKafka::Consumer::OffsetTail(1));
      } else {
        subscription_->set_offset(offset_tail_ - 1);
      }
      err = consumer_->seek(*subscription_, timeout_);
      if (err != RdKafka::ERR_NO_ERROR) {
        return errors::Internal("failed to seek tail -1: ", RdKafka::err2str(err));
      }
      do {
        message.reset(consumer_->consume(timeout_));
      } while (message->err() == RdKafka::ERR__TRANSPORT);
      if (message->err() != RdKafka::ERR_NO_ERROR) {
        return errors::Internal("failed to consume tail message: ", RdKafka::err2str(message->err()));
      }
      offset_tail_ = message->offset();

      subscription_->set_offset(offset_head_);
      err = consumer_->seek(*subscription_, timeout_);
      if (err != RdKafka::ERR_NO_ERROR) {
        return errors::Internal("failed to seek tail -1: ", RdKafka::err2str(err));
      }
    }

    index_ = 0;
    offset_ = subscription_->offset();
    return Status::OK();
  }
  Status Read(const int64 start, const int64 stop, const string& component, int64* record_read, Tensor* value, Tensor* label) override {
    *record_read = 0;
    if (start != index_) {
      // Case 1: start > 0
      if (start > 0) {
        // If EOF has aleady been reached, then we will just return empty
        // in case start > index_
        if (start > index_) {
          // EOF of topic for stream dataset, just return
          if (consumer_.get() == nullptr) {
            return Status::OK();
          }
          // EOF of topic for normal dataset, just return
          if (offset_tail_ >= 0 && offset_ >= offset_tail_) {
            return Status::OK();
          }
        }
        // Otherwise  just return error
        return errors::InvalidArgument("dataset can not seek to a random location");
      }
      // Case 2: start = 0
      // If offset_tail_ < 0, then this is a stream dataset,
      // we should return empty data 
      if (offset_tail_ < 0) {
        return errors::InvalidArgument("stream dataset can not return back to 0");
      }
      // otherwise we seek to the beginning, this is important to make dataset
      // repeatable so that it could be used in training
      subscription_->set_offset(offset_head_);
      RdKafka::ErrorCode err = consumer_->seek((*subscription_), timeout_);
      if (err != RdKafka::ERR_NO_ERROR) {
        return errors::Internal("failed to seek partition: ", RdKafka::err2str(err));
      }
      index_ = 0;
      offset_ = subscription_->offset();
    }

    while (consumer_.get() != nullptr && index_ < stop) {
      if (!kafka_event_cb_.run()) {
        return errors::Internal("failed to consume due to all brokers down");
      }
      if (offset_tail_ >= 0 && offset_ >= offset_tail_) {
        // EOF of topic, reset index to 0
        break;
      }

      std::unique_ptr<RdKafka::Message> message(consumer_->consume(timeout_));
      if (message->err() == RdKafka::ERR_NO_ERROR) {
        // Produce the line as output.
        value->flat<string>()(index_) = std::string(static_cast<const char*>(message->payload()), message->len());
        (*record_read)++;
        index_++;
        offset_ = message->offset();
        continue;
      }
      if (message->err() == RdKafka::ERR__PARTITION_EOF) {
        LOG(INFO) << "Partition reach EOF, current offset: " << subscription_->offset();
        if (offset_tail_ >= 0) {
          return errors::Internal("received EOF unexpected: ", message->errstr());
        }
        // If this is the end of file, then stop here.
        // Also, this means, file could not be reused.
        if (eof_) {
          consumer_.reset(nullptr);
          break;
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
  Status Spec(const string& component, PartialTensorShape* shape, DataType* dtype, bool label) override {
    *shape = PartialTensorShape({-1});
    *dtype = DT_STRING;
    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("KafkaReadable[]");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  int64 index_ GUARDED_BY(mu_);
  int64 offset_ GUARDED_BY(mu_);

  std::unique_ptr<RdKafka::TopicPartition> subscription_ GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::KafkaConsumer> consumer_ GUARDED_BY(mu_);
  KafkaEventCb kafka_event_cb_ = KafkaEventCb();
  int64 timeout_ GUARDED_BY(mu_);
  bool eof_ GUARDED_BY(mu_);

  int64 offset_head_ GUARDED_BY(mu_);
  int64 offset_tail_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>KafkaReadableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<KafkaReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>KafkaReadableRead").Device(DEVICE_CPU),
                        IOReadableReadOp<KafkaReadable>);

class DecodeAvroOp : public OpKernel {
 public:
  explicit DecodeAvroOp(OpKernelConstruction* context) : OpKernel(context) {
   OP_REQUIRES_OK(context, context->GetAttr("schema", &schema_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    avro::ValidSchema avro_schema;
    std::istringstream ss(schema_);
    string error;
    OP_REQUIRES(context, (avro::compileJsonSchema(ss, avro_schema, error)), errors::Unimplemented("Avro schema error: ", error));

    avro::GenericDatum datum(avro_schema);
    std::vector<Tensor*> value;
    value.reserve(avro_schema.root()->names());
    for (size_t i = 0; i < avro_schema.root()->names(); i++) {
      Tensor* value_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(static_cast<int64>(i), input_tensor->shape(), &value_tensor));
      value.push_back(value_tensor);
    }

    for (int64 entry_index = 0; entry_index < input_tensor->NumElements(); entry_index++) {
      const string& entry = input_tensor->flat<string>()(entry_index);
      std::unique_ptr<avro::InputStream> in = avro::memoryInputStream((const uint8_t*)entry.data(), entry.size());

      avro::DecoderPtr d = avro::binaryDecoder();
      d->init(*in);
      avro::decode(*d, datum);
      const avro::GenericRecord& record = datum.value<avro::GenericRecord>();
      for (int i = 0; i < avro_schema.root()->names(); i++) {
        const avro::GenericDatum& field = record.fieldAt(i);
        switch(field.type()) {
        case avro::AVRO_BOOL:
          value[i]->flat<bool>()(entry_index) = field.value<bool>();
          break;
        case avro::AVRO_INT:
          value[i]->flat<int32>()(entry_index) = field.value<int32_t>();
          break;
        case avro::AVRO_LONG:
          value[i]->flat<int64>()(entry_index) = field.value<int64_t>();
          break;
        case avro::AVRO_FLOAT:
          value[i]->flat<float>()(entry_index) = field.value<float>();
          break;
        case avro::AVRO_DOUBLE:
          value[i]->flat<double>()(entry_index) = field.value<double>();
          break;
        case avro::AVRO_STRING:
          value[i]->flat<string>()(entry_index) = field.value<string>();
          break;
        case avro::AVRO_BYTES:
          {
            const std::vector<uint8_t>& field_value = field.value<std::vector<uint8_t>>();
            string v;
            if (field_value.size() > 0) {
              v.resize(field_value.size());
              memcpy(&v[0], &field_value[0], field_value.size());
            }
            value[i]->flat<string>()(entry_index) = std::move(v);
          }
          break;
        case avro::AVRO_FIXED:
          {
            const std::vector<uint8_t>& field_value = field.value<avro::GenericFixed>().value();
            string v;
            if (field_value.size() > 0) {
              v.resize(field_value.size());
              memcpy(&v[0], &field_value[0], field_value.size());
            }
            value[i]->flat<string>()(entry_index) = std::move(v);
          }
          break;
        case avro::AVRO_ENUM:
          value[i]->flat<string>()(entry_index) = field.value<avro::GenericEnum>().symbol();
          break;
        default:
          OP_REQUIRES(context, false, errors::InvalidArgument("unsupported data type: ", field.type()));
        }
      }
    }
  }
private:
  string schema_;
};


REGISTER_KERNEL_BUILDER(Name("IO>DecodeAvro").Device(DEVICE_CPU),
                        DecodeAvroOp);

}  // namespace data
}  // namespace tensorflow
