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

#include "absl/time/clock.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow_io/core/golang_ops.h"

namespace tensorflow {
namespace data {
namespace {
class SeriesReadableResourceBase : public ResourceBase {
 public:
  virtual Status Init(const string& input,
                      const std::vector<string>& metadata) = 0;
  virtual Status Spec(int64* start, int64* stop) = 0;
  virtual Status Peek(const int64 start, const int64 stop,
                      TensorShape* shape) = 0;
  virtual Status Read(const int64 start, const int64 stop, Tensor* timestamp,
                      Tensor* value) = 0;
};

class PrometheusReadableResource : public SeriesReadableResourceBase {
 public:
  PrometheusReadableResource(Env* env) : env_(env) {}
  ~PrometheusReadableResource() {}

  Status Init(const string& input,
              const std::vector<string>& metadata) override {
    mutex_lock l(mu_);

    query_ = input;
    endpoint_ = "http://localhost:9090";
    int64 length = -1;
    int64 offset = -1;
    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find("length=") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2 || !strings::safe_strto64(parts[1], &length)) {
          return errors::InvalidArgument("invalid configuration: ",
                                         metadata[i]);
        }
      } else if (metadata[i].find("offset=") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2 || !strings::safe_strto64(parts[1], &offset)) {
          return errors::InvalidArgument("invalid configuration: ",
                                         metadata[i]);
        }
      } else if (metadata[i].find("endpoint=") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid configuration: ",
                                         metadata[i]);
        }
        endpoint_ = parts[1];
      }
    }
    if (length < 0) {
      return errors::InvalidArgument("length must be provided");
    }
    if (offset < 0) {
      stop_ = absl::GetCurrentTimeNanos() / 1000000;
    } else {
      stop_ = offset;
    }
    start_ = stop_ - length * 1000;
    return Status::OK();
  }
  Status Spec(int64* start, int64* stop) override {
    mutex_lock l(mu_);
    *start = start_;
    *stop = stop_;
    return Status::OK();
  }
  Status Peek(const int64 start, const int64 stop,
              TensorShape* shape) override {
    mutex_lock l(mu_);
    int64 interval = (stop - start) / 1000;
    *shape = TensorShape({interval});
    return Status::OK();
  }
  Status Read(const int64 start, const int64 stop, Tensor* timestamp,
              Tensor* value) override {
    GoString endpoint_go = {endpoint_.c_str(),
                            static_cast<int64>(endpoint_.size())};
    GoString query_go = {query_.c_str(), static_cast<int64>(query_.size())};

    GoSlice timestamp_go = {timestamp->flat<int64>().data(),
                            timestamp->NumElements(), timestamp->NumElements()};
    GoSlice value_go = {value->flat<double>().data(), value->NumElements(),
                        value->NumElements()};

    GoInt returned =
        QueryRange(endpoint_go, query_go, start, stop, timestamp_go, value_go);
    if (returned < 0) {
      return errors::InvalidArgument("unable to query prometheus");
    }

    mutex_lock l(mu_);
    return Status::OK();
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return "PrometheusReadableResource";
  }

 protected:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  string query_ GUARDED_BY(mu_);
  string endpoint_ GUARDED_BY(mu_);
  int64 start_ GUARDED_BY(mu_);
  int64 stop_ GUARDED_BY(mu_);
};

class PrometheusReadableInitOp
    : public ResourceOpKernel<PrometheusReadableResource> {
 public:
  explicit PrometheusReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<PrometheusReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<PrometheusReadableResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<string>()();

    std::vector<string> metadata;
    const Tensor* metadata_tensor;
    OP_REQUIRES_OK(context, context->input("metadata", &metadata_tensor));
    for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
      metadata.push_back(metadata_tensor->flat<string>()(i));
    }

    OP_REQUIRES_OK(context, resource_->Init(input, metadata));
  }
  Status CreateResource(PrometheusReadableResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new PrometheusReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class PrometheusReadableSpecOp : public OpKernel {
 public:
  explicit PrometheusReadableSpecOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    PrometheusReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    int64 start, stop;
    OP_REQUIRES_OK(context, resource->Spec(&start, &stop));

    Tensor* start_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &start_tensor));
    start_tensor->scalar<int64>()() = start;

    Tensor* stop_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}), &stop_tensor));
    stop_tensor->scalar<int64>()() = stop;
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class PrometheusReadableReadOp : public OpKernel {
 public:
  explicit PrometheusReadableReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    PrometheusReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    int64 start = start_tensor->scalar<int64>()();

    const Tensor* stop_tensor;
    OP_REQUIRES_OK(context, context->input("stop", &stop_tensor));
    int64 stop = stop_tensor->scalar<int64>()();

    TensorShape shape;
    OP_REQUIRES_OK(context, resource->Peek(start, stop, &shape));

    Tensor* timestamp_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, shape, &timestamp_tensor));

    Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, shape, &value_tensor));
    if (shape.dim_size(0) > 0) {
      OP_REQUIRES_OK(
          context, resource->Read(start, stop, timestamp_tensor, value_tensor));
    }
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
REGISTER_KERNEL_BUILDER(Name("IO>PrometheusReadableInit").Device(DEVICE_CPU),
                        PrometheusReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>PrometheusReadableSpec").Device(DEVICE_CPU),
                        PrometheusReadableSpecOp);
REGISTER_KERNEL_BUILDER(Name("IO>PrometheusReadableRead").Device(DEVICE_CPU),
                        PrometheusReadableReadOp);

class PrometheusScrapeOp : public OpKernel {
 public:
  explicit PrometheusScrapeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* metric_tensor;
    OP_REQUIRES_OK(context, context->input("metric", &metric_tensor));
    const string& metric = metric_tensor->scalar<string>()();

    const Tensor* endpoint_tensor;
    OP_REQUIRES_OK(context, context->input("endpoint", &endpoint_tensor));
    const string& endpoint = endpoint_tensor->scalar<string>()();

    Tensor* timestamp_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                     &timestamp_tensor));

    Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}), &value_tensor));

    GoString metric_go = {metric.c_str(), static_cast<int64>(metric.size())};
    GoString endpoint_go = {endpoint.c_str(),
                            static_cast<int64>(endpoint.size())};
    GoSlice value_go = {value_tensor->flat<double>().data(), 1, 1};

    GoInt returned = Scrape(endpoint_go, metric_go, value_go);
    if (returned == 0) {
      timestamp_tensor->scalar<int64>()() =
          absl::GetCurrentTimeNanos() / 1000000;
    }
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
REGISTER_KERNEL_BUILDER(Name("IO>PrometheusScrape").Device(DEVICE_CPU),
                        PrometheusScrapeOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
