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

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/grpc/server.grpc.pb.h"
#include <grpc++/grpc++.h>

namespace tensorflow {
namespace data {

class GRPCIOServerImplementation : public IOIterableInterface {
  class GRPCIOServerServiceState {
   public:
    GRPCIOServerServiceState(grpc::ServerReader<Request>* reader)
     : reader_(reader) {}
    ~GRPCIOServerServiceState() {}
    grpc::ServerReader<Request>* Reader() {
      return reader_;
    }
    void Wait() {
      mutex_lock l(mu_);
      condition_variable_.wait(l);
    }
    void Notify() {
      condition_variable_.notify_all();
    }
   private:
    mutable mutex mu_;
    condition_variable condition_variable_;
    grpc::ServerReader<Request>* reader_;
  };
  class GRPCIOServerServiceImpl final : public GRPCIOServer::Service {
   public:
    virtual grpc::Status Next(grpc::ServerContext* context, grpc::ServerReader<Request>* reader, Response* response) override {
      const std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
      string component = "";
      for (auto iter = metadata.begin(); iter != metadata.end(); ++iter) {
        string key = string(iter->first.data(), iter->first.size());
        if (key == "component") {
          component = string(iter->second.data(), iter->second.size());
        }
      }
      if (component == "") {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "component key not provided");
      }
      if (state_.find(component) != state_.end()) {
        string message = "component key already exist: " + component;
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, message);
      }
      state_[component] = std::unique_ptr<GRPCIOServerServiceState>(new GRPCIOServerServiceState(reader));
      state_[component]->Wait();
      return grpc::Status::OK;
    }
    GRPCIOServerServiceState* State(const string& component) {
      return state_[component].get();
    }
   private:
    std::unordered_map<string, std::unique_ptr<GRPCIOServerServiceState>> state_;
  };

 public:
  GRPCIOServerImplementation(Env* env)
  : env_(env) {}

  ~GRPCIOServerImplementation() {
    server_->Shutdown();
  }
  Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) override {
    std::string server_address("0.0.0.0:50051");

    builder_.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder_.RegisterService(&service_);
    server_ = builder_.BuildAndStart();

    LOG(INFO) << "GRPC IO Server listening on " << server_address << std::endl;
    thread_pool_.reset(new thread::ThreadPool(env_, "GRPCIOServer", 1));
    thread_pool_->Schedule([this] {
      server_->Wait();
    });
    return Status::OK();
  }
  Status Next(const int64 capacity, const Tensor& component, Tensor* tensor, int64* record_read) override {
    *record_read = 0;
    Request request;
    while (service_.State(component.scalar<string>()())->Reader()->Read(&request)) {
      tensor->flat<string>()((*record_read)) = request.item();
      (*record_read)++;
      if ((*record_read) >= capacity) {
        return Status::OK();
      }
    }
    service_.State(component.scalar<string>()())->Notify();
    return Status::OK();
  }
  Status Spec(const Tensor& component, PartialTensorShape* shape, DataType* dtype) override {
    *shape = PartialTensorShape({-1});
    *dtype = DT_STRING;
    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("GRPCIOServer[]");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  GRPCIOServerServiceImpl service_;
  grpc::ServerBuilder builder_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

REGISTER_KERNEL_BUILDER(Name("GRPCIOServerInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<GRPCIOServerImplementation>);
REGISTER_KERNEL_BUILDER(Name("GRPCIOServerIterableNext").Device(DEVICE_CPU),
                        IOIterableNextOp<GRPCIOServerImplementation>);

}  // namespace data
}  // namespace tensorflow
