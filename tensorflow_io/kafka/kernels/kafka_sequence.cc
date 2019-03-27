/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/public/version.h"
#include "kernels/sequence_ops.h"

#include "rdkafkacpp.h"
#include <deque>

namespace tensorflow {

class KafkaOutputSequence : public OutputSequence {
 public:
  KafkaOutputSequence(Env* env)
   : OutputSequence(env) {}

  virtual ~KafkaOutputSequence() override {
    Flush();
  }
  virtual Status Flush() override {
    if (producer_.get() != nullptr) {
      RdKafka::ErrorCode err = producer_->flush(timeout_);
      if (!(err == RdKafka::ERR_NO_ERROR)) {
        return errors::Internal("Failed to flush message:", RdKafka::err2str(err));
      }
    }
    return Status::OK();
  }
  virtual Status Output() override {
    if (fifo_.front().get() != nullptr) {
      while (fifo_.size() != 0 && fifo_.front().get() != nullptr) {
        RdKafka::ErrorCode err = producer_->produce(
            topic_.get(), partition_, RdKafka::Producer::RK_MSG_COPY,
            const_cast<char *>(fifo_.front().get()->c_str()), fifo_.front().get()->size(),
            NULL, NULL);
        if (!(err == RdKafka::ERR_NO_ERROR)) {
          return errors::Internal("Failed to produce message:", RdKafka::err2str(err));
	}

        fifo_.pop_front();
        base_++;
      }
    }
    return Status::OK();
  }
#if TF_MAJOR_VERSION==1&&TF_MINOR_VERSION==13
  virtual string DebugString() {
#else
  virtual string DebugString() const {
#endif
    return strings::StrCat("KafkaOutputSequence[]");
  }
  Status Initialize(const string& topic_str, const string& servers, int32 partition) {
    partition_ = partition;

    std::unique_ptr<RdKafka::Conf> conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    std::unique_ptr<RdKafka::Conf> topic_conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));

    std::string errstr;

    RdKafka::Conf::ConfResult result =
        conf->set("default_topic_conf", topic_conf.get(), errstr);
    if (!(result == RdKafka::Conf::CONF_OK)) {
      return errors::Internal("Failed to set default_topic_conf:", errstr);
    }

    result = conf->set("bootstrap.servers", servers, errstr);
    if (!(result == RdKafka::Conf::CONF_OK)) {
      return errors::Internal("Failed to set bootstrap.servers ", servers, ":", errstr);
    }

    producer_.reset(RdKafka::Producer::create(conf.get(), errstr));
    if (!(producer_.get() != nullptr)) {
      return errors::Internal("Failed to create producer:", errstr);
    }

    topic_.reset(RdKafka::Topic::create(producer_.get(), topic_str, topic_conf.get(), errstr));
    if (!(topic_.get() != nullptr)) {
      return errors::Internal("Failed to create topic ", topic_str, ":", errstr);
    }

    return Status::OK();
  }
 private:
  int32 partition_ GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::Producer> producer_ GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::Topic> topic_ GUARDED_BY(mu_);
  static const int timeout_ = 5000;
};

class KafkaOutputSequenceOp : public OutputSequenceOp<KafkaOutputSequence> {
 public:
  explicit KafkaOutputSequenceOp(OpKernelConstruction* context)
    : OutputSequenceOp<KafkaOutputSequence>(context) {
  }
 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<KafkaOutputSequence>::Compute(context);
    mutex_lock l(mu_);
    const Tensor* topic_tensor;
    const Tensor* servers_tensor;
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

    OP_REQUIRES_OK(context, resource_->Initialize(topic_str, servers, partition));
  }
};


REGISTER_KERNEL_BUILDER(Name("KafkaOutputSequence").Device(DEVICE_CPU),
                        KafkaOutputSequenceOp);


REGISTER_KERNEL_BUILDER(Name("KafkaOutputSequenceSetItem").Device(DEVICE_CPU),
                        OutputSequenceSetItemOp<KafkaOutputSequence>);

REGISTER_KERNEL_BUILDER(Name("KafkaOutputSequenceFlush").Device(DEVICE_CPU),
                        OutputSequenceFlushOp<KafkaOutputSequence>);
}  // namespace tensorflow
