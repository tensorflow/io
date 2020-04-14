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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/core/kernels/sequence_ops.h"

#include "api/Compiler.hh"
#include "api/DataFile.hh"
#include "api/Generic.hh"
#include "api/Stream.hh"
#include "api/Validator.hh"
#include "rdkafkacpp.h"

#include <deque>
#include <unordered_map>
#include "rdkafkacpp.h"

namespace tensorflow {

class KafkaDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* topics_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("topics", &topics_tensor));
    OP_REQUIRES(
        ctx, topics_tensor->dims() <= 1,
        errors::InvalidArgument("`topics` must be a scalar or a vector."));

    std::vector<string> topics;
    topics.reserve(topics_tensor->NumElements());
    for (int i = 0; i < topics_tensor->NumElements(); ++i) {
      topics.push_back(topics_tensor->flat<tstring>()(i));
    }

    tstring servers = "";
    OP_REQUIRES_OK(
        ctx, data::ParseScalarArgument<tstring>(ctx, "servers", &servers));
    tstring group = "";
    OP_REQUIRES_OK(
        ctx, data::ParseScalarArgument<tstring>(ctx, "group", &group));
    bool eof = false;
    OP_REQUIRES_OK(ctx, data::ParseScalarArgument<bool>(ctx, "eof", &eof));
    int64 timeout = -1;
    OP_REQUIRES_OK(ctx,
                   data::ParseScalarArgument<int64>(ctx, "timeout", &timeout));
    OP_REQUIRES(ctx, (timeout > 0),
                errors::InvalidArgument(
                    "Timeout value should be large than 0, got ", timeout));

    const Tensor* config_global_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("config_global", &config_global_tensor));
    std::vector<string> config_global;
    config_global.reserve(config_global_tensor->NumElements());
    for (int i = 0; i < config_global_tensor->NumElements(); ++i) {
      config_global.push_back(config_global_tensor->flat<tstring>()(i));
    }

    const Tensor* config_topic_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("config_topic", &config_topic_tensor));
    std::vector<string> config_topic;
    config_topic.reserve(config_topic_tensor->NumElements());
    for (int i = 0; i < config_topic_tensor->NumElements(); ++i) {
      config_topic.push_back(config_topic_tensor->flat<tstring>()(i));
    }
    bool message_key = false;
    OP_REQUIRES_OK(
        ctx, data::ParseScalarArgument<bool>(ctx, "message_key", &message_key));

    *output = new Dataset(ctx, std::move(topics), servers, group, eof, timeout,
                          std::move(config_global), std::move(config_topic),
                          message_key);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<string> topics,
            const string& servers, const string& group, const bool eof,
            const int64 timeout, std::vector<string> config_global,
            std::vector<string> config_topic, const bool message_key)
        : DatasetBase(DatasetContext(ctx)),
          topics_(std::move(topics)),
          servers_(servers),
          group_(group),
          eof_(eof),
          timeout_(timeout),
          config_global_(std::move(config_global)),
          config_topic_(std::move(config_topic)),
          message_key_(message_key) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Kafka")}));
    }

    const DataTypeVector& output_dtypes() const override {
      if (message_key_) {
        static DataTypeVector* dtypes =
            new DataTypeVector({DT_STRING, DT_STRING});
        return *dtypes;
      }
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      if (message_key_) {
        static std::vector<PartialTensorShape>* shapes =
            new std::vector<PartialTensorShape>({{}, {}});
        return *shapes;
      }
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override { return "KafkaDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* topics = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(topics_, &topics));
      Node* servers = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(servers_, &servers));
      Node* group = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(group_, &group));
      Node* eof = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(eof_, &eof));
      Node* timeout = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(timeout_, &timeout));
      Node* config_global = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(config_global_, &config_global));
      Node* config_topic = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(config_topic_, &config_topic));
      Node* message_key = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(message_key_, &message_key));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this,
                        {topics, servers, group, eof, timeout, config_global,
                         config_topic, message_key},
                        output));
      return Status::OK();
    }

   private:
    class KafkaEventCb : public RdKafka::EventCb {
     public:
      KafkaEventCb(bool& run) : run_(run) {}

      void event_cb(RdKafka::Event& event) {
        switch (event.type()) {
          case RdKafka::Event::EVENT_ERROR:
            LOG(ERROR) << "EVENT_ERROR: "
                       << "(" << RdKafka::err2str(event.err())
                       << "): " << event.str();
            { run_ = !event.fatal(); }
            break;

          case RdKafka::Event::EVENT_STATS:
            LOG(ERROR) << "EVENT_STATS: " << event.str();
            break;

          case RdKafka::Event::EVENT_LOG:
            LOG(ERROR) << "EVENT_LOG: " << event.severity() << "-"
                       << event.fac().c_str() << "-" << event.str().c_str();
            break;

          case RdKafka::Event::EVENT_THROTTLE:
            LOG(ERROR) << "EVENT_THROTTLE: " << event.throttle_time()
                       << "ms by " << event.broker_name() << " id "
                       << (int)event.broker_id();
            break;

          default:
            LOG(ERROR) << "EVENT: " << event.type() << " ("
                       << RdKafka::err2str(event.err()) << "): " << event.str();
            break;
        }
      }

     private:
      bool& run_;
    };

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          // We are currently processing a topic, so try to read the next line.
          if (consumer_.get()) {
            while (run_) {
              if (limit_ >= 0 &&
                  (topic_partition_->offset() >= limit_ || offset_ >= limit_)) {
                // EOF current topic
                break;
              }
              std::unique_ptr<RdKafka::Message> message(
                  consumer_->consume(dataset()->timeout_));
              if (message->err() == RdKafka::ERR_NO_ERROR) {
                // Produce the line as output.
                Tensor line_tensor(cpu_allocator(), DT_STRING, {});
                line_tensor.scalar<tstring>()() =
                    std::string(static_cast<const char*>(message->payload()),
                                message->len());
                out_tensors->emplace_back(std::move(line_tensor));
                if (dataset()->message_key_) {
                  Tensor key_tensor(cpu_allocator(), DT_STRING, {});
                  if (message->key() != nullptr) {
                    key_tensor.scalar<tstring>()() = string(*message->key());
                  } else {
                    key_tensor.scalar<tstring>()() = "";
                  }
                  out_tensors->emplace_back(std::move(key_tensor));
                }
                *end_of_sequence = false;
                // Sync offset
                offset_ = message->offset();
                return Status::OK();
              }

              if (message->err() == RdKafka::ERR__PARTITION_EOF) {
                LOG(INFO) << "Partition reach EOF: "
                          << dataset()->topics_[current_topic_index_]
                          << ", current offset: " << offset_;

                if (dataset()->eof_) break;
              } else if (message->err() == RdKafka::ERR__TRANSPORT) {
                // Not return error here because consumer will try re-connect.
                LOG(ERROR) << "Broker transport failure: " << message->errstr();
              } else if (message->err() != RdKafka::ERR__TIMED_OUT) {
                LOG(ERROR) << "Failed to consume: " << message->errstr();
                return errors::Internal("Failed to consume: ",
                                        message->errstr());
              }

              message.reset(nullptr);
            }

            if (!run_) {
              return errors::Internal(
                  "Failed to consume due to all brokers down");
            }

            // We have reached the end of the current topic, so maybe
            // move on to next topic.
            ResetStreamsLocked();
            ++current_topic_index_;
          }

          // Iteration ends when there are no more topic to process.
          if (current_topic_index_ == dataset()->topics_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }

          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        } while (true);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("current_topic_index"),
                                               current_topic_index_));

        // `consumer_` is empty if
        // 1. GetNext has not been called even once.
        // 2. All topics have been read and iterator has been exhausted.
        if (consumer_.get()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("current_pos"), offset_));

          LOG(INFO) << "Save current topic: "
                    << dataset()->topics_[current_topic_index_]
                    << ", current offset: " << offset_;
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        ResetStreamsLocked();
        int64 current_topic_index;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("current_topic_index"),
                                              &current_topic_index));
        current_topic_index_ = size_t(current_topic_index);
        // The key "current_pos" is written only if the iterator was saved
        // with an open topic.
        if (reader->Contains(full_name("current_pos"))) {
          int64 current_pos;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("current_pos"), &current_pos));

          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
          topic_partition_->set_offset(current_pos);
          if (topic_partition_->offset() != current_pos) {
            return errors::Internal("Failed to restore to offset ",
                                    current_pos);
          }

          std::vector<RdKafka::TopicPartition*> partitions;
          partitions.emplace_back(topic_partition_.get());
          RdKafka::ErrorCode err = consumer_->assign(partitions);
          if (err != RdKafka::ERR_NO_ERROR) {
            return errors::Internal(
                "Failed to assign partition [", topic_partition_->topic(), ", ",
                topic_partition_->partition(), ", ", topic_partition_->offset(),
                "]:", RdKafka::err2str(err));
          }
          offset_ = current_pos;

          LOG(INFO) << "Restore to topic: "
                    << "[" << topic_partition_->topic() << ":"
                    << topic_partition_->partition() << ":"
                    << topic_partition_->offset() << "]";
        }
        return Status::OK();
      }

     private:
      // Sets up Kafka streams to read from the topic at
      // `current_topic_index_`.
      Status SetupStreamsLocked(Env* env) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_topic_index_ >= dataset()->topics_.size()) {
          return errors::InvalidArgument(
              "current_topic_index_:", current_topic_index_,
              " >= topics_.size():", dataset()->topics_.size());
        }

        // Actually move on to next topic.
        string entry = dataset()->topics_[current_topic_index_];

        std::vector<string> parts = str_util::Split(entry, ":");
        if (parts.size() < 1) {
          return errors::InvalidArgument("Invalid parameters: ", entry);
        }
        string topic = parts[0];
        int32 partition = 0;
        if (parts.size() > 1) {
          if (!strings::safe_strto32(parts[1], &partition)) {
            return errors::InvalidArgument("Invalid parameters: ", entry);
          }
        }
        int64 offset = 0;
        if (parts.size() > 2) {
          if (!strings::safe_strto64(parts[2], &offset)) {
            return errors::InvalidArgument("Invalid parameters: ", entry);
          }
        }

        topic_partition_.reset(
            RdKafka::TopicPartition::create(topic, partition, offset));

        offset_ = topic_partition_->offset();
        limit_ = -1;
        if (parts.size() > 3) {
          if (!strings::safe_strto64(parts[3], &limit_)) {
            return errors::InvalidArgument("Invalid parameters: ", entry);
          }
        }

        std::unique_ptr<RdKafka::Conf> conf(
            RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
        std::unique_ptr<RdKafka::Conf> topic_conf(
            RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));
        RdKafka::Conf::ConfResult result = RdKafka::Conf::CONF_UNKNOWN;

        std::string errstr;

        for (auto it = dataset()->config_topic_.begin();
             it != dataset()->config_topic_.end(); it++) {
          std::vector<string> parts = str_util::Split(*it, "=");
          if (parts.size() != 2) {
            return errors::InvalidArgument("Invalid topic configuration: ",
                                           *it);
          }
          result = topic_conf->set(parts[0], parts[1], errstr);
          if (result != RdKafka::Conf::CONF_OK) {
            return errors::Internal("Failed to do topic configuration:", *it,
                                    "error:", errstr);
          }
          LOG(INFO) << "Kafka topic configuration: " << *it;
        }

        result = conf->set("default_topic_conf", topic_conf.get(), errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("Failed to set default_topic_conf:", errstr);
        }

        for (auto it = dataset()->config_global_.begin();
             it != dataset()->config_global_.end(); it++) {
          std::vector<string> parts = str_util::Split(*it, "=");
          if (parts.size() != 2) {
            return errors::InvalidArgument("Invalid global configuration: ",
                                           *it);
          }
          result = conf->set(parts[0], parts[1], errstr);
          if (result != RdKafka::Conf::CONF_OK) {
            return errors::Internal("Failed to do global configuration: ", *it,
                                    "error:", errstr);
          }
          LOG(INFO) << "Kafka global configuration: " << *it;
        }

        result = conf->set("event_cb", &kafka_event_cb, errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("Failed to set event_cb:", errstr);
        }

        result = conf->set("bootstrap.servers", dataset()->servers_, errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("Failed to set bootstrap.servers ",
                                  dataset()->servers_, ":", errstr);
        }
        result = conf->set("group.id", dataset()->group_, errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("Failed to set group.id ", dataset()->group_,
                                  ":", errstr);
        }

        // Always enable.partition.eof=true
        result = conf->set("enable.partition.eof", "true", errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("Failed to set enable.partition.eof=true",
                                  ":", errstr);
        }

        consumer_.reset(RdKafka::KafkaConsumer::create(conf.get(), errstr));
        if (!consumer_.get()) {
          return errors::Internal("Failed to create consumer:", errstr);
        }

        std::vector<RdKafka::TopicPartition*> partitions;
        partitions.emplace_back(topic_partition_.get());
        RdKafka::ErrorCode err = consumer_->assign(partitions);
        if (err != RdKafka::ERR_NO_ERROR) {
          return errors::Internal(
              "Failed to assign partition [", topic_partition_->topic(), ", ",
              topic_partition_->partition(), ", ", topic_partition_->offset(),
              "]:", RdKafka::err2str(err));
        }
        LOG(INFO) << "Kafka stream starts with current offset: "
                  << topic_partition_->offset();

        return Status::OK();
      }

      // Resets all Kafka streams.
      void ResetStreamsLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (consumer_.get()) {
          consumer_->unassign();
          consumer_->close();
          consumer_.reset(nullptr);
        }
      }

      mutex mu_;
      bool run_ TF_GUARDED_BY(mu_) = true;
      size_t current_topic_index_ TF_GUARDED_BY(mu_) = 0;
      int64 offset_ TF_GUARDED_BY(mu_) = 0;
      int64 limit_ TF_GUARDED_BY(mu_) = -1;
      std::unique_ptr<RdKafka::TopicPartition> topic_partition_ TF_GUARDED_BY(mu_);
      std::unique_ptr<RdKafka::KafkaConsumer> consumer_ TF_GUARDED_BY(mu_);
      KafkaEventCb kafka_event_cb = KafkaEventCb(run_);
    };

    const std::vector<string> topics_;
    const tstring servers_;
    const tstring group_;
    const bool eof_;
    const int64 timeout_;
    const std::vector<string> config_global_;
    const std::vector<string> config_topic_;
    const bool message_key_;
  };
};

class WriteKafkaOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  void Compute(OpKernelContext* context) override {
    const Tensor* message_tensor;
    const Tensor* topic_tensor;
    const Tensor* servers_tensor;
    OP_REQUIRES_OK(context, context->input("message", &message_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(message_tensor->shape()),
                errors::InvalidArgument(
                    "Message tensor must be scalar, but had shape: ",
                    message_tensor->shape().DebugString()));
    OP_REQUIRES_OK(context, context->input("topic", &topic_tensor));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(topic_tensor->shape()),
        errors::InvalidArgument("Topic tensor must be scalar, but had shape: ",
                                topic_tensor->shape().DebugString()));
    OP_REQUIRES_OK(context, context->input("servers", &servers_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(servers_tensor->shape()),
                errors::InvalidArgument(
                    "Servers tensor must be scalar, but had shape: ",
                    servers_tensor->shape().DebugString()));

    const string& message = message_tensor->scalar<tstring>()();
    const string& topic_string = topic_tensor->scalar<tstring>()();
    std::vector<string> parts = str_util::Split(topic_string, ":");
    OP_REQUIRES(context, (parts.size() >= 1),
                errors::InvalidArgument("Invalid parameters: ", topic_string));

    const string& topic_str = parts[0];
    int32 partition = 0;
    if (parts.size() > 1) {
      OP_REQUIRES(
          context, !strings::safe_strto32(parts[1], &partition),
          errors::InvalidArgument("Invalid parameters: ", topic_string));
    }

    const string& servers = servers_tensor->scalar<tstring>()();

    std::unique_ptr<RdKafka::Conf> conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    std::unique_ptr<RdKafka::Conf> topic_conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));

    std::string errstr;

    RdKafka::Conf::ConfResult result =
        conf->set("default_topic_conf", topic_conf.get(), errstr);
    OP_REQUIRES(context, (result == RdKafka::Conf::CONF_OK),
                errors::Internal("Failed to set default_topic_conf:", errstr));

    result = conf->set("bootstrap.servers", servers, errstr);
    OP_REQUIRES(context, (result == RdKafka::Conf::CONF_OK),
                errors::Internal("Failed to set bootstrap.servers ", servers,
                                 ":", errstr));

    std::unique_ptr<RdKafka::Producer> producer(
        RdKafka::Producer::create(conf.get(), errstr));
    OP_REQUIRES(context, producer.get() != nullptr,
                errors::Internal("Failed to create producer:", errstr));

    std::unique_ptr<RdKafka::Topic> topic(RdKafka::Topic::create(
        producer.get(), topic_str, topic_conf.get(), errstr));
    OP_REQUIRES(
        context, topic.get() != nullptr,
        errors::Internal("Failed to create topic ", topic_str, ":", errstr));

    RdKafka::ErrorCode err = producer->produce(
        topic.get(), partition, RdKafka::Producer::RK_MSG_COPY,
        const_cast<char*>(message.c_str()), message.size(), NULL, NULL);
    OP_REQUIRES(
        context, (err == RdKafka::ERR_NO_ERROR),
        errors::Internal("Failed to produce message:", RdKafka::err2str(err)));

    err = producer->flush(timeout_);
    OP_REQUIRES(
        context, (err == RdKafka::ERR_NO_ERROR),
        errors::Internal("Failed to flush message:", RdKafka::err2str(err)));
    context->set_output(0, context->input(0));
  }

 private:
  static const int timeout_ = 5000;
};

class KafkaOutputSequence : public OutputSequence {
 public:
  KafkaOutputSequence(Env* env) : OutputSequence(env) {}

  virtual ~KafkaOutputSequence() override { Flush(); }
  virtual Status Flush() override {
    if (producer_.get() != nullptr) {
      RdKafka::ErrorCode err = producer_->flush(timeout_);
      if (!(err == RdKafka::ERR_NO_ERROR)) {
        return errors::Internal("Failed to flush message:",
                                RdKafka::err2str(err));
      }
    }
    return Status::OK();
  }
  virtual Status Output() override {
    if (fifo_.front().get() != nullptr) {
      while (fifo_.size() != 0 && fifo_.front().get() != nullptr) {
        RdKafka::ErrorCode err = producer_->produce(
            topic_.get(), partition_, RdKafka::Producer::RK_MSG_COPY,
            const_cast<char*>(fifo_.front().get()->c_str()),
            fifo_.front().get()->size(), NULL, NULL);
        if (!(err == RdKafka::ERR_NO_ERROR)) {
          return errors::Internal("Failed to produce message:",
                                  RdKafka::err2str(err));
        }

        fifo_.pop_front();
        base_++;
      }
    }
    return Status::OK();
  }
  virtual string DebugString() const {
    return strings::StrCat("KafkaOutputSequence[]");
  }
  Status Initialize(const string& topic_str, int32 partition,
                    const std::vector<string>& metadata) {
    partition_ = partition;

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

    // producer.properties:
    //   bootstrap.servers=localhost:9092
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

    topic_.reset(RdKafka::Topic::create(producer_.get(), topic_str,
                                        conf_topic.get(), errstr));
    if (!(topic_.get() != nullptr)) {
      return errors::Internal("Failed to create topic ", topic_str, ":",
                              errstr);
    }

    return Status::OK();
  }

 private:
  int32 partition_ TF_GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::Producer> producer_ TF_GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::Topic> topic_ TF_GUARDED_BY(mu_);
  static const int timeout_ = 5000;
};

class KafkaOutputSequenceOp : public OutputSequenceOp<KafkaOutputSequence> {
 public:
  explicit KafkaOutputSequenceOp(OpKernelConstruction* context)
      : OutputSequenceOp<KafkaOutputSequence>(context) {}

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<KafkaOutputSequence>::Compute(context);
    mutex_lock l(mu_);

    const Tensor* topic_tensor;
    OP_REQUIRES_OK(context, context->input("topic", &topic_tensor));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(topic_tensor->shape()),
        errors::InvalidArgument("Topic tensor must be scalar, but had shape: ",
                                topic_tensor->shape().DebugString()));

    const Tensor* metadata_tensor;
    OP_REQUIRES_OK(context, context->input("metadata", &metadata_tensor));
    std::vector<string> metadata;
    for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
      metadata.push_back(metadata_tensor->flat<tstring>()(i));
    }

    const string& topic_string = topic_tensor->scalar<tstring>()();
    std::vector<string> parts = str_util::Split(topic_string, ":");
    OP_REQUIRES(context, (parts.size() >= 1),
                errors::InvalidArgument("Invalid parameters: ", topic_string));

    const string& topic_str = parts[0];
    int32 partition = 0;
    if (parts.size() > 1) {
      OP_REQUIRES(
          context, !strings::safe_strto32(parts[1], &partition),
          errors::InvalidArgument("Invalid parameters: ", topic_string));
    }

    OP_REQUIRES_OK(context,
                   resource_->Initialize(topic_str, partition, metadata));
  }
};

namespace data {

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
  bool run_ TF_GUARDED_BY(mu_) = true;
};

class DecodeAvroResource : public ResourceBase {
 public:
  DecodeAvroResource(Env* env) : env_(env) {}
  ~DecodeAvroResource() {}

  Status Init(const string& input) {
    mutex_lock lock(mu_);
    schema_ = input;
    schema_stream_ = std::istringstream(schema_);

    string error;
    if (!(avro::compileJsonSchema(schema_stream_, avro_schema_, error))) {
      return errors::Unimplemented("Avro schema error: ", error);
    }

    return Status::OK();
  }
  const avro::ValidSchema& avro_schema() { return avro_schema_; }
  string DebugString() const override { return "DecodeAvroResource"; }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  string schema_ TF_GUARDED_BY(mu_);
  std::istringstream schema_stream_;
  avro::ValidSchema avro_schema_;
};

class DecodeAvroOp : public OpKernel {
 public:
  explicit DecodeAvroOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    DecodeAvroResource* resource;
    std::unique_ptr<DecodeAvroResource> resource_scope;
    if (context->input_dtype(1) == DT_RESOURCE) {
      OP_REQUIRES_OK(context,
                     GetResourceFromContext(context, "schema", &resource));
    } else {
      const Tensor* schema_tensor;
      OP_REQUIRES_OK(context, context->input("schema", &schema_tensor));
      const string& schema = schema_tensor->scalar<tstring>()();

      resource_scope.reset(new DecodeAvroResource(env_));
      OP_REQUIRES_OK(context, resource_scope->Init(schema));
      resource_scope->Ref();
      resource = resource_scope.get();
    }
    core::ScopedUnref unref(resource);

    std::vector<Tensor*> value;
    value.reserve(resource->avro_schema().root()->names());
    for (size_t i = 0; i < resource->avro_schema().root()->names(); i++) {
      Tensor* value_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(static_cast<int64>(i),
                                                       input_tensor->shape(),
                                                       &value_tensor));
      value.push_back(value_tensor);
    }

    avro::GenericDatum datum(resource->avro_schema());
    for (int64 entry_index = 0; entry_index < input_tensor->NumElements();
         entry_index++) {
      const string& entry = input_tensor->flat<tstring>()(entry_index);
      std::unique_ptr<avro::InputStream> in =
          avro::memoryInputStream((const uint8_t*)entry.data(), entry.size());

      avro::DecoderPtr d = avro::binaryDecoder();
      d->init(*in);
      avro::decode(*d, datum);
      const avro::GenericRecord& record = datum.value<avro::GenericRecord>();
      for (int i = 0; i < resource->avro_schema().root()->names(); i++) {
        const avro::GenericDatum& field = record.fieldAt(i);
        switch (field.type()) {
          case avro::AVRO_NULL:
            switch (context->expected_output_dtype(i)) {
              case DT_BOOL:
                value[i]->flat<bool>()(entry_index) = false;
                break;
              case DT_INT32:
                value[i]->flat<int32>()(entry_index) = 0;
                break;
              case DT_INT64:
                value[i]->flat<int64>()(entry_index) = 0;
                break;
              case DT_FLOAT:
                value[i]->flat<float>()(entry_index) = 0.0;
                break;
              case DT_DOUBLE:
                value[i]->flat<double>()(entry_index) = 0.0;
                break;
              case DT_STRING:
                value[i]->flat<tstring>()(entry_index) = "";
                break;
              default:
                OP_REQUIRES(context, false,
                            errors::InvalidArgument(
                                "unsupported data type against AVRO_NULL: ",
                                field.type()));
            }
            break;
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
          case avro::AVRO_STRING: {
            // make a concrete explicit copy as otherwise avro may override the
            // underlying buffer.
            const string& field_value = field.value<string>();
            string v;
            if (field_value.size() > 0) {
              v.resize(field_value.size());
              memcpy(&v[0], &field_value[0], field_value.size());
            }
            value[i]->flat<tstring>()(entry_index) = v;
          } break;
          case avro::AVRO_BYTES: {
            const std::vector<uint8_t>& field_value =
                field.value<std::vector<uint8_t>>();
            string v;
            if (field_value.size() > 0) {
              v.resize(field_value.size());
              memcpy(&v[0], &field_value[0], field_value.size());
            }
            value[i]->flat<tstring>()(entry_index) = std::move(v);
          } break;
          case avro::AVRO_FIXED: {
            const std::vector<uint8_t>& field_value =
                field.value<avro::GenericFixed>().value();
            string v;
            if (field_value.size() > 0) {
              v.resize(field_value.size());
              memcpy(&v[0], &field_value[0], field_value.size());
            }
            value[i]->flat<tstring>()(entry_index) = std::move(v);
          } break;
          case avro::AVRO_ENUM:
            value[i]->flat<tstring>()(entry_index) =
                field.value<avro::GenericEnum>().symbol();
            break;
          default:
            OP_REQUIRES(context, false,
                        errors::InvalidArgument("unsupported data type: ",
                                                field.type()));
        }
      }
    }
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class EncodeAvroOp : public OpKernel {
 public:
  explicit EncodeAvroOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("schema", &schema_));
  }

  void Compute(OpKernelContext* context) override {
    // Make sure input have the same elements;
    for (int64 i = 1; i < context->num_inputs(); i++) {
      OP_REQUIRES(
          context,
          (context->input(0).NumElements() == context->input(i).NumElements()),
          errors::InvalidArgument("number of elements different: input 0 (",
                                  context->input(0).NumElements(),
                                  ") vs. input ", i, " (",
                                  context->input(i).NumElements(), ")"));
    }

    avro::ValidSchema avro_schema;
    std::istringstream ss(schema_);
    string error;
    OP_REQUIRES(context, (avro::compileJsonSchema(ss, avro_schema, error)),
                errors::Unimplemented("Avro schema error: ", error));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, context->input(0).shape(), &output_tensor));

    for (int64 entry_index = 0; entry_index < context->input(0).NumElements();
         entry_index++) {
      std::ostringstream ss;
      std::unique_ptr<avro::OutputStream> out = avro::ostreamOutputStream(ss);

      avro::GenericDatum datum(avro_schema);
      avro::GenericRecord& record = datum.value<avro::GenericRecord>();
      for (int i = 0; i < avro_schema.root()->names(); i++) {
        switch (record.fieldAt(i).type()) {
          case avro::AVRO_BOOL:
            record.setFieldAt(i,
                              avro::GenericDatum(
                                  context->input(i).flat<bool>()(entry_index)));
            break;
          case avro::AVRO_INT:
            record.setFieldAt(
                i, avro::GenericDatum(static_cast<int32_t>(
                       context->input(i).flat<int32>()(entry_index))));
            break;
          case avro::AVRO_LONG:
            record.setFieldAt(
                i, avro::GenericDatum(static_cast<int64_t>(
                       context->input(i).flat<int64>()(entry_index))));
            break;
          case avro::AVRO_FLOAT:
            record.setFieldAt(
                i, avro::GenericDatum(
                       context->input(i).flat<float>()(entry_index)));
            break;
          case avro::AVRO_DOUBLE:
            record.setFieldAt(
                i, avro::GenericDatum(
                       context->input(i).flat<double>()(entry_index)));
            break;
          case avro::AVRO_STRING: {
            // make a concrete explicit copy as otherwise avro may override the
            // underlying buffer?? happens in decode (not verified in encode
            // yet).
            const string& v = context->input(i).flat<tstring>()(entry_index);
            string field_value;
            field_value.resize(v.size());
            if (field_value.size() > 0) {
              memcpy(&field_value[0], &v[0], field_value.size());
            }
            record.setFieldAt(i, avro::GenericDatum(field_value));
          } break;
          case avro::AVRO_BYTES: {
            const string& v = context->input(i).flat<tstring>()(entry_index);
            std::vector<uint8_t> field_value;
            field_value.resize(v.size());
            if (field_value.size() > 0) {
              memcpy(&field_value[0], &v[0], field_value.size());
            }
            record.setFieldAt(i, avro::GenericDatum(field_value));
          } break;
          case avro::AVRO_FIXED:
          case avro::AVRO_ENUM:
          default:
            OP_REQUIRES(context, false,
                        errors::InvalidArgument("unsupported data type: ",
                                                record.fieldAt(i).type()));
        }
      }
      avro::EncoderPtr e = avro::binaryEncoder();
      e->init(*out);
      avro::encode(*e, datum);
      out->flush();
      output_tensor->flat<tstring>()(entry_index) = ss.str();
    }
  }

 private:
  string schema_;
};
class DecodeAvroInitOp : public ResourceOpKernel<DecodeAvroResource> {
 public:
  explicit DecodeAvroInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<DecodeAvroResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<DecodeAvroResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    OP_REQUIRES_OK(context, resource_->Init(input_tensor->scalar<tstring>()()));
  }
  Status CreateResource(DecodeAvroResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new DecodeAvroResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>KafkaDataset").Device(DEVICE_CPU),
                        KafkaDatasetOp);

REGISTER_KERNEL_BUILDER(Name("IO>WriteKafka").Device(DEVICE_CPU), WriteKafkaOp);

REGISTER_KERNEL_BUILDER(Name("IO>KafkaOutputSequence").Device(DEVICE_CPU),
                        KafkaOutputSequenceOp);
REGISTER_KERNEL_BUILDER(
    Name("IO>KafkaOutputSequenceSetItem").Device(DEVICE_CPU),
    OutputSequenceSetItemOp<KafkaOutputSequence>);
REGISTER_KERNEL_BUILDER(Name("IO>KafkaOutputSequenceFlush").Device(DEVICE_CPU),
                        OutputSequenceFlushOp<KafkaOutputSequence>);

REGISTER_KERNEL_BUILDER(Name("IO>KafkaDecodeAvro").Device(DEVICE_CPU),
                        DecodeAvroOp);
REGISTER_KERNEL_BUILDER(Name("IO>KafkaEncodeAvro").Device(DEVICE_CPU),
                        EncodeAvroOp);
REGISTER_KERNEL_BUILDER(Name("IO>KafkaDecodeAvroInit").Device(DEVICE_CPU),
                        DecodeAvroInitOp);

}  // namespace data
}  // namespace tensorflow
