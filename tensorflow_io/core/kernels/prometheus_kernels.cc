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

class PrometheusReadableResource : public ResourceBase {
 public:
  PrometheusReadableResource(Env* env) : env_(env) {}
  ~PrometheusReadableResource() {}

  Status Init(const string& input, const std::vector<string>& metadata,
              std::function<Status(const TensorShape& shape, Tensor** metrics)>
                  allocate_func) {
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

    GoString endpoint_go = {endpoint_.c_str(),
                            static_cast<int64>(endpoint_.size())};
    GoString query_go = {query_.c_str(), static_cast<int64>(query_.size())};

    GoSlice jobs_go = {nullptr, 0, 0};
    GoSlice instances_go = {nullptr, 0, 0};
    GoSlice names_go = {nullptr, 0, 0};
    GoInt returned = QuerySpecs(endpoint_go, query_go, start_, jobs_go,
                                instances_go, names_go);
    if (returned < 0) {
      return errors::InvalidArgument("unable to query prometheus");
    }
    Tensor* metrics;
    TF_RETURN_IF_ERROR(allocate_func(TensorShape({returned, 3}), &metrics));

    // The buffer is used to hold memory in C++ and pass to Golang
    // Mamory management are done in C++
    // TODO: Much of the logic could be pushed to Golang by passing
    // Tensor directly.
    std::vector<string> buffer;
    std::vector<GoSlice> jobs_v;
    for (GoInt i = 0; i < returned; i++) {
      buffer.push_back(string());
      buffer.back().resize(1024);
      jobs_v.push_back(GoSlice{&buffer.back()[0], 1024 - 1, 1024 - 1});
    }
    std::vector<GoSlice> instances_v;
    for (GoInt i = 0; i < returned; i++) {
      buffer.push_back(string());
      buffer.back().resize(1024);
      instances_v.push_back(GoSlice{&buffer.back()[0], 1024 - 1, 1024 - 1});
    }
    std::vector<GoSlice> names_v;
    for (GoInt i = 0; i < returned; i++) {
      buffer.push_back(string());
      buffer.back().resize(1024);
      names_v.push_back(GoSlice{&buffer.back()[0], 1024 - 1, 1024 - 1});
    }
    jobs_go.data = &jobs_v[0];
    jobs_go.len = returned;
    jobs_go.cap = returned;
    instances_go.data = &instances_v[0];
    instances_go.len = returned;
    instances_go.cap = returned;
    names_go.data = &names_v[0];
    names_go.len = returned;
    names_go.cap = returned;
    returned = QuerySpecs(endpoint_go, query_go, start_, jobs_go, instances_go,
                          names_go);
    if (returned < 0) {
      return errors::InvalidArgument("unable to query prometheus");
    }
    for (size_t index = 0; index < returned; index++) {
      string job((char*)(jobs_v[index].data));
      string instance((char*)(instances_v[index].data));
      string name((char*)(names_v[index].data));
      jobs_.push_back(job);
      instances_.push_back(instance);
      names_.push_back(name);
      metrics->tensor<tstring, 2>()(index, 0) = job;
      metrics->tensor<tstring, 2>()(index, 1) = instance;
      metrics->tensor<tstring, 2>()(index, 2) = name;
    }
    return Status::OK();
  }
  Status Spec(int64* start, int64* stop) {
    mutex_lock l(mu_);
    *start = start_;
    *stop = stop_;
    return Status::OK();
  }
  Status Read(const int64 start, const int64 stop, std::vector<string>& jobs,
              std::vector<string>& instances, std::vector<string>& names,
              std::function<Status(const TensorShape& timestamp_shape,
                                   const TensorShape& value_shape,
                                   Tensor** timestamp, Tensor** value)>
                  allocate_func) {
    mutex_lock l(mu_);
    int64 interval = (stop - start) / 1000;

    if (jobs.size() != instances.size() || jobs.size() != names.size()) {
      return errors::InvalidArgument(
          "jobs, instances, names must be equal: ", jobs.size(), " vs. ",
          instances.size(), " vs. ", names.size());
    }

    Tensor* value;
    Tensor* timestamp;
    TF_RETURN_IF_ERROR(
        allocate_func(TensorShape({interval}),
                      TensorShape({static_cast<int64>(names.size()), interval}),
                      &timestamp, &value));

    GoString endpoint_go = {endpoint_.c_str(),
                            static_cast<int64>(endpoint_.size())};
    GoString query_go = {query_.c_str(), static_cast<int64>(query_.size())};

    for (size_t index = 0; index < jobs.size(); index++) {
      GoString job_go = {jobs[index].data(),
                         static_cast<ptrdiff_t>(jobs[index].size())};
      GoString instance_go = {instances[index].data(),
                              static_cast<ptrdiff_t>(instances[index].size())};
      GoString name_go = {names[index].data(),
                          static_cast<ptrdiff_t>(names[index].size())};
      GoSlice timestamp_go = {timestamp->flat<int64>().data(),
                              timestamp->NumElements(),
                              timestamp->NumElements()};
      double* value_p = (double*)(value->flat<double>().data()) +
                        index * timestamp->NumElements();
      GoSlice value_go = {value_p, timestamp->NumElements(),
                          timestamp->NumElements()};
      GoInt returned = QueryRange(endpoint_go, query_go, start, stop, job_go,
                                  instance_go, name_go, timestamp_go, value_go);
      if (returned < 0) {
        return errors::InvalidArgument("unable to query prometheus");
      }
    }

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
  std::vector<string> jobs_ GUARDED_BY(mu_);
  std::vector<string> instances_ GUARDED_BY(mu_);
  std::vector<string> names_ GUARDED_BY(mu_);
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
    string input = input_tensor->scalar<tstring>()();

    std::vector<string> metadata;
    const Tensor* metadata_tensor;
    OP_REQUIRES_OK(context, context->input("metadata", &metadata_tensor));
    for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
      metadata.push_back(metadata_tensor->flat<tstring>()(i));
    }

    OP_REQUIRES_OK(
        context,
        resource_->Init(
            input, metadata,
            [&](const TensorShape& shape, Tensor** metrics) -> Status {
              TF_RETURN_IF_ERROR(context->allocate_output(1, shape, metrics));
              return Status::OK();
            }));
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
    const int64 start = start_tensor->scalar<int64>()();

    const Tensor* stop_tensor;
    OP_REQUIRES_OK(context, context->input("stop", &stop_tensor));
    const int64 stop = stop_tensor->scalar<int64>()();

    const Tensor* metrics_tensor;
    OP_REQUIRES_OK(context, context->input("metrics", &metrics_tensor));
    std::vector<string> jobs, instances, names;
    for (int64 i = 0; i < metrics_tensor->NumElements() / 3; i++) {
      jobs.push_back(metrics_tensor->tensor<tstring, 2>()(i, 0));
      instances.push_back(metrics_tensor->tensor<tstring, 2>()(i, 1));
      names.push_back(metrics_tensor->tensor<tstring, 2>()(i, 2));
    }

    OP_REQUIRES_OK(
        context,
        resource->Read(start, stop, jobs, instances, names,
                       [&](const TensorShape& timestamp_shape,
                           const TensorShape& value_shape, Tensor** timestamp,
                           Tensor** value) -> Status {
                         TF_RETURN_IF_ERROR(context->allocate_output(
                             0, timestamp_shape, timestamp));
                         TF_RETURN_IF_ERROR(
                             context->allocate_output(1, value_shape, value));
                         return Status::OK();
                       }));
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
    string metric = metric_tensor->scalar<tstring>()();

    const Tensor* endpoint_tensor;
    OP_REQUIRES_OK(context, context->input("endpoint", &endpoint_tensor));
    string endpoint = endpoint_tensor->scalar<tstring>()();

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
