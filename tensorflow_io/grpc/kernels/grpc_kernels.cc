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
    GRPCIOServerServiceState(grpc::ServerReader<Request>* reader, const PartialTensorShape& shape, const DataType& dtype, const PartialTensorShape& label_shape, const DataType& label_dtype)
     : reader_(reader)
     , shape_(shape)
     , dtype_(dtype)
     , label_shape_(label_shape)
     , label_dtype_(label_dtype) {}

    ~GRPCIOServerServiceState() {}
    grpc::ServerReader<Request>* Reader() {
      return reader_;
    }
    Status Spec(PartialTensorShape* shape, DataType* dtype) {
      *shape = shape_;
      *dtype = dtype_;
      return Status::OK();
    }
    Status Label(PartialTensorShape* shape, DataType* dtype) {
      if (label_dtype_ == DT_INVALID) {
        return errors::Unimplemented("Label");
      } 
      *shape = label_shape_;
      *dtype = label_dtype_;
      return Status::OK();
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
    PartialTensorShape shape_;
    DataType dtype_;
    PartialTensorShape label_shape_;
    DataType label_dtype_;
  };
  class GRPCIOServerServiceImpl final : public GRPCIOServer::Service {
   public:
    virtual grpc::Status Next(grpc::ServerContext* context, grpc::ServerReader<Request>* reader, Response* response) override {

      LOG(INFO) << "GRPC IO Server received request" << std::endl;
      const std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
      string component = "";
      string spec = "";
      string label = "";
      for (auto iter = metadata.begin(); iter != metadata.end(); ++iter) {
        string key = string(iter->first.data(), iter->first.size());
        string val = string(iter->second.data(), iter->second.size());
        if (key == "component") {
          component = val;
        } else if (key == "spec") {
          spec = val;
        } else if (key == "label") {
          label = val;
        }
      }
      if (component == "") {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "component key not provided");
      }
      if (spec == "") {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "spec key not provided");
      }

      std::vector<std::string> entries;

      PartialTensorShape shape;
      DataType dtype = DT_INVALID;

      entries = absl::StrSplit(spec, ":");

      DataTypeFromString(entries[1], &dtype);

      std::vector<int64> dims;
      entries = absl::StrSplit(entries[0], ",");
      for (size_t i = 0; i < entries.size(); i++) {
        int64 dim = 0;
        absl::SimpleAtoi<int64>(entries[i], &dim);
        dims.push_back(dim);
      }
      shape = PartialTensorShape(dims);

      PartialTensorShape label_shape;
      DataType label_dtype = DT_INVALID;
      if (label != "") {
        entries = absl::StrSplit(label, ":");

        DataTypeFromString(entries[1], &label_dtype);

        std::vector<int64> dims;
        entries = absl::StrSplit(entries[0], ",");
        for (size_t i = 0; i < entries.size(); i++) {
          int64 dim = 0;
          absl::SimpleAtoi<int64>(entries[i], &dim);
          dims.push_back(dim);
        }
        label_shape = PartialTensorShape(dims);
      }

      if (state_.find(component) != state_.end()) {
        string message = "component key already exist: " + component;
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, message);
      }
      state_[component] = std::unique_ptr<GRPCIOServerServiceState>(new GRPCIOServerServiceState(reader, shape, dtype, label_shape, label_dtype));
      LOG(INFO) << "GRPC IO Server request wait for processing" << std::endl;
      state_[component]->Wait();
      return grpc::Status::OK;
    }
    GRPCIOServerServiceState* State(const string& component) {
      std::unordered_map<string, std::unique_ptr<GRPCIOServerServiceState>>::iterator lookup = state_.find(component);
      if (lookup == state_.end()) {
        return nullptr;
      }
      return lookup->second.get();
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
  Status Next(const int64 capacity, const Tensor& component, int64* record_read, Tensor* value, Tensor* label) override {
    GRPCIOServerServiceState* p = service_.State(component.scalar<string>()());
    if (p == nullptr) {
      return errors::InvalidArgument("unable to find component: ", component.scalar<string>()());
    }
    *record_read = 0;
    Request request;
    while (p->Reader()->Read(&request)) {
      TensorProto value_field;
      request.value().UnpackTo(&value_field);
      value->FromProto(value_field);
      if (request.has_label()) {
        TensorProto label_field;
        request.label().UnpackTo(&label_field);
        label->FromProto(label_field);
      }
      (*record_read)++;
      if ((*record_read) >= capacity) {
        return Status::OK();
      }
    }
    p->Notify();
    return Status::OK();
  }
  Status Spec(const Tensor& component, PartialTensorShape* shape, DataType* dtype) override {
    GRPCIOServerServiceState* p = service_.State(component.scalar<string>()());
    if (p == nullptr) {
      return errors::InvalidArgument("unable to find component: ", component.scalar<string>()());
    }
    return p->Spec(shape, dtype);
  }
  Status Label(const Tensor& component, PartialTensorShape* shape, DataType* dtype) override {
    GRPCIOServerServiceState* p = service_.State(component.scalar<string>()());
    if (p == nullptr) {
      return errors::InvalidArgument("unable to find component: ", component.scalar<string>()());
    }
    return p->Label(shape, dtype);
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
