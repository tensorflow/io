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

#include "tensorflow/core/framework/dataset.h"

#include <grpc++/grpc++.h>

#include "google/pubsub/v1/pubsub.grpc.pb.h"

namespace tensorflow {

using grpc::ClientContext;
using google::pubsub::v1::Subscriber;
using google::pubsub::v1::PullRequest;
using google::pubsub::v1::PullResponse;
using google::pubsub::v1::AcknowledgeRequest;

class PubSubDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* subscriptions_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("subscriptions", &subscriptions_tensor));
    OP_REQUIRES(
        ctx, subscriptions_tensor->dims() <= 1,
        errors::InvalidArgument("`subscriptions` must be a scalar or a vector."));

    std::vector<string> subscriptions;
    subscriptions.reserve(subscriptions_tensor->NumElements());
    for (int i = 0; i < subscriptions_tensor->NumElements(); ++i) {
      subscriptions.push_back(subscriptions_tensor->flat<string>()(i));
    }

    std::string server = "";
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<std::string>(ctx, "server", &server));
    bool eof = false;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "eof", &eof));
    int64 timeout = -1;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "timeout", &timeout));
    OP_REQUIRES(ctx, (timeout > 0),
                errors::InvalidArgument(
                    "Timeout value should be large than 0, got ", timeout));
    *output = new Dataset(ctx, std::move(subscriptions), server, eof, timeout);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<string> subscriptions,
            const string& server, const bool eof, const int64 timeout)
        : DatasetBase(DatasetContext(ctx)),
          subscriptions_(std::move(subscriptions)),
          server_(server),
          eof_(eof),
          timeout_(timeout) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::PubSub")}));
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

    string DebugString() const override { return "PubSubDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* subscriptions = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(subscriptions_, &subscriptions));
      Node* server = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(server_, &server));
      Node* eof = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(eof_, &eof));
      Node* timeout = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(timeout_, &timeout));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {subscriptions, server, eof, timeout}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          // We are currently processing a subscription, so try to read the next line.
          if (stub_.get()) {
            ClientContext context;
            if (dataset()->timeout_ > 0) {
              std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(dataset()->timeout_);
              context.set_deadline(deadline);
            }
            while (true) {
              string subscription = dataset()->subscriptions_[current_subscription_index_];
              PullRequest request;
              request.set_subscription(subscription);
              request.set_max_messages(1);
              PullResponse response;
              auto status = stub_->Pull(&context, request, &response);
              if (!status.ok()) {
                return errors::Internal("Failed to receive message: ", status.error_message());
	      }
              if (status.ok() && response.received_messages().size() == 0 && dataset()->eof_) {
                // EOF current subscription
                break;
              }
              if (status.ok() && response.received_messages().size() != 0) {
                // Produce the line as output.
                Tensor line_tensor(cpu_allocator(), DT_STRING, {});
                line_tensor.scalar<string>()() =
                    std::string((response.received_messages(0).message().data()));
                out_tensors->emplace_back(std::move(line_tensor));
                *end_of_sequence = false;
                // Acknowledge
                AcknowledgeRequest acknowledge;
                acknowledge.add_ack_ids(response.received_messages(0).ack_id());
                acknowledge.set_subscription(subscription);
		google::protobuf::Empty empty;
		ClientContext ack_context;
                status = stub_->Acknowledge(&ack_context, acknowledge, &empty);
                return Status::OK();
	      }
            }

            // We have reached the end of the current subscription, so maybe
            // move on to next subscription.
            ResetStreamsLocked();
            ++current_subscription_index_;
          }

          // Iteration ends when there are no more subscription to process.
          if (current_subscription_index_ == dataset()->subscriptions_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }

          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        } while (true);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        return errors::Unimplemented("SaveInternal is currently not supported");
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return errors::Unimplemented(
            "RestoreInternal is currently not supported");
      }

     private:
      // Sets up PubSub streams to read from the subscription at
      // `current_subscription_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_subscription_index_ >= dataset()->subscriptions_.size()) {
          return errors::InvalidArgument(
              "current_subscription_index_:", current_subscription_index_,
              " >= subscriptions_.size():", dataset()->subscriptions_.size());
        }

        // Actually move on to next subscription.
        string subscription = dataset()->subscriptions_[current_subscription_index_];
        string server = dataset()->server_;
        auto creds = grpc::GoogleDefaultCredentials();
        if (dataset()->server_.find("http://") == 0) {
          server = dataset()->server_.substr(7);
          creds = grpc::InsecureChannelCredentials();
        } else if (dataset()->server_.find("https://") == 0) {
          // https://pubsub.googleapis.com
          server = dataset()->server_.substr(8);
        }
        stub_ = Subscriber::NewStub(grpc::CreateChannel(server, creds));

        return Status::OK();
      }

      // Resets all PubSub streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        stub_.reset(nullptr);
      }

      mutex mu_;
      size_t current_subscription_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr< Subscriber::Stub> stub_ GUARDED_BY(mu_);
    };

    const std::vector<string> subscriptions_;
    const std::string server_;
    const bool eof_;
    const int64 timeout_;
  };
};

REGISTER_KERNEL_BUILDER(Name("PubSubDataset").Device(DEVICE_CPU),
                        PubSubDatasetOp);

}  // namespace tensorflow
