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
#include "go/prometheus.h"

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
}  // namespace data
}  // namespace tensorflow
