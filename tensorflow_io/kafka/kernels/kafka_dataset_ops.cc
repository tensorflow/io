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
#include "tensorflow/core/platform/logging.h"

#include "rdkafkacpp.h"

namespace tensorflow {
namespace data {

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
      topics.push_back(topics_tensor->flat<string>()(i));
    }

    std::string servers = "";
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<std::string>(ctx, "servers", &servers));
    std::string group = "";
    OP_REQUIRES_OK(ctx, ParseScalarArgument<std::string>(ctx, "group", &group));
    bool eof = false;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "eof", &eof));
    int64 timeout = -1;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "timeout", &timeout));
    OP_REQUIRES(ctx, (timeout > 0),
                errors::InvalidArgument(
                    "Timeout value should be large than 0, got ", timeout));

    const Tensor* config_global_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("config_global", &config_global_tensor));
    std::vector<string> config_global;
    config_global.reserve(config_global_tensor->NumElements());
    for (int i = 0; i < config_global_tensor->NumElements(); ++i) {
      config_global.push_back(config_global_tensor->flat<string>()(i));
    }

    const Tensor* config_topic_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("config_topic", &config_topic_tensor));
    std::vector<string> config_topic;
    config_topic.reserve(config_topic_tensor->NumElements());
    for (int i = 0; i < config_topic_tensor->NumElements(); ++i) {
      config_topic.push_back(config_topic_tensor->flat<string>()(i));
    }

    *output = new Dataset(ctx, std::move(topics), servers, group, eof, timeout,
                          std::move(config_global), std::move(config_topic));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<string> topics,
            const string& servers, const string& group, const bool eof,
            const int64 timeout, std::vector<string> config_global,
            std::vector<string> config_topic)
        : DatasetBase(DatasetContext(ctx)),
          topics_(std::move(topics)),
          servers_(servers),
          group_(group),
          eof_(eof),
          timeout_(timeout),
          config_global_(std::move(config_global)),
          config_topic_(std::move(config_topic)) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Kafka")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
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
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {topics, servers, group, eof, timeout,
                        config_global, config_topic}, output));
      return Status::OK();
    }

   private:
      class KafkaEventCb : public RdKafka::EventCb {
        public:
          KafkaEventCb(bool &run):run_(run) {}

          void event_cb (RdKafka::Event &event) {
            switch (event.type())
            {
              case RdKafka::Event::EVENT_ERROR:
                LOG(ERROR) << "EVENT_ERROR: " << "(" <<
                  RdKafka::err2str(event.err()) << "): " << event.str();
                if (event.err() == RdKafka::ERR__ALL_BROKERS_DOWN)
                  run_ = false;
                break;

              case RdKafka::Event::EVENT_STATS:
                LOG(ERROR) << "EVENT_STATS: " << event.str();
                break;

              case RdKafka::Event::EVENT_LOG:
                LOG(ERROR) << "EVENT_LOG: " << event.severity() <<
                  "-" << event.fac().c_str() << "-" << event.str().c_str();
                break;

              case RdKafka::Event::EVENT_THROTTLE:
                LOG(ERROR) << "EVENT_THROTTLE: " << event.throttle_time() << "ms by "
                  << event.broker_name() << " id " << (int)event.broker_id();
                break;

              default:
                LOG(ERROR) << "EVENT: " << event.type() <<
                  " (" << RdKafka::err2str(event.err()) << "): " << event.str();
                break;
            }
          }

       private:
        bool &run_;
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
                line_tensor.scalar<string>()() =
                    std::string(static_cast<const char*>(message->payload()),
                                message->len());
                out_tensors->emplace_back(std::move(line_tensor));
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
              }
              else if (message->err() == RdKafka::ERR__TRANSPORT) {
                  // Not return error here because consumer will try re-connect.
                  LOG(ERROR) << "Broker transport failure: " << message->errstr();
              }
              else if (message->err() != RdKafka::ERR__TIMED_OUT) {
                  LOG(ERROR) << "Failed to consume: " << message->errstr();
                  return errors::Internal("Failed to consume: ",
                                          message->errstr());
              }

              message.reset(nullptr);
            }

            if (!run_) {
              return errors::Internal("Failed to consume due to all brokers down");
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

          LOG(INFO) << "Save current topic: " << dataset()->topics_[current_topic_index_]
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

          LOG(INFO) << "Restore to topic: " << "["
            << topic_partition_->topic() << ":" << topic_partition_->partition()
            << ":" << topic_partition_->offset() << "]";

        }
        return Status::OK();
      }

     private:
      // Sets up Kafka streams to read from the topic at
      // `current_topic_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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
          it != dataset()->config_topic_.end(); it++)
        {
          std::vector<string> parts = str_util::Split(*it, "=");
          if (parts.size() != 2) {
            return errors::InvalidArgument("Invalid topic configuration: ", *it);
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
          it != dataset()->config_global_.end(); it++)
        {
          std::vector<string> parts = str_util::Split(*it, "=");
          if (parts.size() != 2) {
            return errors::InvalidArgument("Invalid global configuration: ", *it);
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
        LOG(INFO) << "Kafka stream starts with current offset: " << topic_partition_->offset();

        return Status::OK();
      }

      // Resets all Kafka streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (consumer_.get()) {
          consumer_->unassign();
          consumer_->close();
          consumer_.reset(nullptr);
        }
      }

      mutex mu_;
      bool run_ GUARDED_BY(mu_) = true;
      size_t current_topic_index_ GUARDED_BY(mu_) = 0;
      int64 offset_ GUARDED_BY(mu_) = 0;
      int64 limit_ GUARDED_BY(mu_) = -1;
      std::unique_ptr<RdKafka::TopicPartition> topic_partition_ GUARDED_BY(mu_);
      std::unique_ptr<RdKafka::KafkaConsumer> consumer_ GUARDED_BY(mu_);
      KafkaEventCb kafka_event_cb = KafkaEventCb(run_);
    };

    const std::vector<string> topics_;
    const std::string servers_;
    const std::string group_;
    const bool eof_;
    const int64 timeout_;
    const std::vector<string> config_global_;
    const std::vector<string> config_topic_;
  };
};

REGISTER_KERNEL_BUILDER(Name("IO>KafkaDataset").Device(DEVICE_CPU),
                        KafkaDatasetOp);

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
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(topic_tensor->shape()),
                errors::InvalidArgument(
                    "Topic tensor must be scalar, but had shape: ",
                    topic_tensor->shape().DebugString()));
    OP_REQUIRES_OK(context, context->input("servers", &servers_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(servers_tensor->shape()),
                errors::InvalidArgument(
                    "Servers tensor must be scalar, but had shape: ",
                    servers_tensor->shape().DebugString()));

    const string& message = message_tensor->scalar<string>()();
    const string& topic_string = topic_tensor->scalar<string>()();
    std::vector<string> parts = str_util::Split(topic_string, ":");
    OP_REQUIRES(context, (parts.size() >= 1),
        errors::InvalidArgument("Invalid parameters: ", topic_string));

    const string& topic_str = parts[0];
    int32 partition = 0;
    if (parts.size() > 1) {
      OP_REQUIRES(context, !strings::safe_strto32(parts[1], &partition),
          errors::InvalidArgument("Invalid parameters: ", topic_string));
    }

    const string& servers = servers_tensor->scalar<string>()();

    std::unique_ptr<RdKafka::Conf> conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    std::unique_ptr<RdKafka::Conf> topic_conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));

    std::string errstr;

    RdKafka::Conf::ConfResult result =
        conf->set("default_topic_conf", topic_conf.get(), errstr);
    OP_REQUIRES(context,  (result == RdKafka::Conf::CONF_OK),
        errors::Internal("Failed to set default_topic_conf:", errstr));

    result = conf->set("bootstrap.servers", servers, errstr);
    OP_REQUIRES(context, (result == RdKafka::Conf::CONF_OK),
        errors::Internal("Failed to set bootstrap.servers ", servers, ":", errstr));

    std::unique_ptr<RdKafka::Producer> producer(RdKafka::Producer::create(conf.get(), errstr));
    OP_REQUIRES(context, producer.get() != nullptr,
        errors::Internal("Failed to create producer:", errstr));

    std::unique_ptr<RdKafka::Topic> topic(RdKafka::Topic::create(producer.get(), topic_str, topic_conf.get(), errstr));
    OP_REQUIRES(context, topic.get() != nullptr,
        errors::Internal("Failed to create topic ", topic_str, ":", errstr));

    RdKafka::ErrorCode err = producer->produce(topic.get(), partition,
                          RdKafka::Producer::RK_MSG_COPY,
                          const_cast<char *>(message.c_str()), message.size(),
                          NULL, NULL);
    OP_REQUIRES(context, (err == RdKafka::ERR_NO_ERROR),
        errors::Internal("Failed to produce message:", RdKafka::err2str(err)));

    err = producer->flush(timeout_);
    OP_REQUIRES(context, (err == RdKafka::ERR_NO_ERROR),
        errors::Internal("Failed to flush message:", RdKafka::err2str(err)));
    context->set_output(0, context->input(0));
  }
private:
  static const int timeout_ = 5000;
};

REGISTER_KERNEL_BUILDER(Name("IO>WriteKafka").Device(DEVICE_CPU), WriteKafkaOp);

}  // namespace data
}  // namespace tensorflow
