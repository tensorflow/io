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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/core/golang_ops.h"

namespace tensorflow {
namespace data {
namespace {


class ReadPrometheusOp : public OpKernel {
 public:
  explicit ReadPrometheusOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& endpoint_tensor = context->input(0);
    const string& endpoint = endpoint_tensor.scalar<string>()();

    const Tensor& query_tensor = context->input(1);
    const string& query = query_tensor.scalar<string>()();

    int64 ts = time(NULL);

    GoString endpoint_go = {endpoint.c_str(), static_cast<int64>(endpoint.size())};
    GoString query_go = {query.c_str(), static_cast<int64>(query.size())};

    GoSlice timestamp_go = {0, 0, 0};
    GoSlice value_go = {0, 0, 0};

    GoInt returned = Query(endpoint_go, query_go, ts, timestamp_go, value_go);
    OP_REQUIRES(context, returned >= 0, errors::InvalidArgument("unable to query prometheus"));

    TensorShape output_shape({static_cast<int64>(returned)});

    Tensor* timestamp_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &timestamp_tensor));
    Tensor* value_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &value_tensor));

    if (returned > 0) {
      timestamp_go.data = timestamp_tensor->flat<int64>().data();
      timestamp_go.len = returned;
      timestamp_go.cap = returned;
      value_go.data = value_tensor->flat<double>().data();
      value_go.len = returned;
      value_go.cap = returned;

      returned = Query(endpoint_go, query_go, ts, timestamp_go, value_go);
      OP_REQUIRES(context, returned >= 0, errors::InvalidArgument("unable to query prometheus to get the value"));
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("ReadPrometheus").Device(DEVICE_CPU),
                        ReadPrometheusOp);


}  // namespace


class PrometheusIndexable : public IOIndexableInterface {
 public:
  PrometheusIndexable(Env* env)
  : env_(env) {}

  ~PrometheusIndexable() {}
  Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) override {
    if (input.size() > 1) {
      return errors::InvalidArgument("more than 1 query is not supported");
    }
    const string& query = input[0];

    string endpoint = "http://localhost:9090";
    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find_first_of("endpoint: ") == 0) {
        endpoint = metadata[i].substr(8);
      }
    }

    int64 ts = time(NULL);

    GoString endpoint_go = {endpoint.c_str(), static_cast<int64>(endpoint.size())};
    GoString query_go = {query.c_str(), static_cast<int64>(query.size())};

    GoSlice timestamp_go = {0, 0, 0};
    GoSlice value_go = {0, 0, 0};

    GoInt returned = Query(endpoint_go, query_go, ts, timestamp_go, value_go);
    if (returned < 0) {
      return errors::InvalidArgument("unable to query prometheus");
    }

    timestamp_.resize(returned);
    value_.resize(returned);

    if (returned > 0) {
      timestamp_go.data = &timestamp_[0];
      timestamp_go.len = returned;
      timestamp_go.cap = returned;
      value_go.data = &value_[0];
      value_go.len = returned;
      value_go.cap = returned;

      returned = Query(endpoint_go, query_go, ts, timestamp_go, value_go);
      if (returned < 0) {
        return errors::InvalidArgument("unable to query prometheus to get the value");
      }
    }

    // timestamp, value
    dtypes_.emplace_back(DT_INT64);
    shapes_.emplace_back(TensorShape({static_cast<int64>(returned)}));
    dtypes_.emplace_back(DT_DOUBLE);
    shapes_.emplace_back(TensorShape({static_cast<int64>(returned), 1}));

    return Status::OK();
  }
  Status Spec(const Tensor& component, PartialTensorShape* shape, DataType* dtype, bool label) override {
    int64 column_index;
    if (component.scalar<string>()() == "index") {
      column_index = 0;
    } else if (component.scalar<string>()() == "value") {
      column_index = 1;
    } else {
      return errors::InvalidArgument("component ", component.scalar<string>()() , " is not supported");
    }

    *shape = shapes_[column_index];
    *dtype = dtypes_[column_index];
    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop, const Tensor& component, Tensor* value, Tensor* label) override {
    if (component.scalar<string>()() == "index") {
      memcpy(&value->flat<int64>().data()[0], &timestamp_[start], sizeof(int64) * (stop - start));
    } else if (component.scalar<string>()() == "value") {
      memcpy(&value->flat<double>().data()[0], &value_[start], sizeof(double) * (stop - start));
    } else {
      return errors::InvalidArgument("component ", component.scalar<string>()() , " is not supported");
    }

    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("PrometheusIndexable");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);

  std::vector<DataType> dtypes_;
  std::vector<TensorShape> shapes_;

  std::vector<int64> timestamp_;
  std::vector<double> value_;
};

REGISTER_KERNEL_BUILDER(Name("PrometheusIndexableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<PrometheusIndexable>);
REGISTER_KERNEL_BUILDER(Name("PrometheusIndexableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<PrometheusIndexable>);
REGISTER_KERNEL_BUILDER(Name("PrometheusIndexableRead").Device(DEVICE_CPU),
                        IOIndexableReadOp<PrometheusIndexable>);

}  // namespace data
}  // namespace tensorflow
