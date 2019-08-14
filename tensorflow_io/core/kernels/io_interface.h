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

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {

class IOInterface : public ResourceBase {
 public:
  virtual Status Init(const string& input, const void* memory_data, const int64 memory_size, const string& metadata) = 0;
  virtual Status Spec(std::vector<DataType>& dtypes, std::vector<PartialTensorShape>& shapes) = 0;

  virtual Status Extra(std::vector<Tensor>* extra) {
    // This is the chance to provide additional extra information which should be appended to extra.
    return Status::OK();
  }
};

class IOIndexableInterface : public IOInterface {
 public:
  virtual Status GetItem(const int64 start, const int64 stop, const int64 step, std::vector<Tensor>& tensors) = 0;
};

template<typename Type>
class IOInterfaceInitOp : public ResourceOpKernel<Type> {
 public:
  explicit IOInterfaceInitOp<Type>(OpKernelConstruction* context)
      : ResourceOpKernel<Type>(context) {
    env_ = context->env();
  }
 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<Type>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<string>()();

    Status status;

    const void *memory_data = nullptr;
    size_t memory_size = 0;

    const Tensor* memory_tensor;
    status = context->input("memory", &memory_tensor);
    if (status.ok()) {
      memory_data = memory_tensor->scalar<string>()().data();
      memory_size = memory_tensor->scalar<string>()().size();
    }

    string metadata;
    const Tensor* metadata_tensor;
    status = context->input("metadata", &metadata_tensor);
    if (status.ok()) {
      metadata = metadata_tensor->scalar<string>()();
    }

    OP_REQUIRES_OK(context, this->resource_->Init(input, memory_data, memory_size, metadata));

    std::vector<DataType> dtypes;
    std::vector<PartialTensorShape> shapes;
    OP_REQUIRES_OK(context, this->resource_->Spec(dtypes, shapes));
    int64 maxrank = 0;
    for (size_t component = 0; component < shapes.size(); component++) {
      for (int64 i = 0; i < shapes[component].dims(); i++) {
        OP_REQUIRES(context, (shapes[component].dim_size(i) > 0), errors::InvalidArgument("component (", component, ")'s shape[", i, "] should not be None, received: ", shapes[component]));
      }
      maxrank = maxrank > shapes[component].dims() ? maxrank : shapes[component].dims();
    }
    Tensor dtypes_tensor(DT_INT64, TensorShape({static_cast<int64>(dtypes.size())}));
    for (size_t i = 0; i < dtypes.size(); i++) {
      dtypes_tensor.flat<int64>()(i) = dtypes[i];
    }
    Tensor shapes_tensor(DT_INT64, TensorShape({static_cast<int64>(dtypes.size()), maxrank}));
    for (size_t component = 0; component < shapes.size(); component++) {
      for (int64 i = 0; i < shapes[component].dims(); i++) {
        shapes_tensor.tensor<int64, 2>()(component, i) = shapes[component].dim_size(i);
      }
      for (int64 i = shapes[component].dims(); i < maxrank; i++) {
        shapes_tensor.tensor<int64, 2>()(component, i) = 0;
      }
    }
    context->set_output(1, dtypes_tensor);
    context->set_output(2, shapes_tensor);

    std::vector<Tensor> extra;
    OP_REQUIRES_OK(context, this->resource_->Extra(&extra));
    for (size_t i = 0; i < extra.size(); i++) {
      context->set_output(3 + i, extra[i]);
    }
  }
  Status CreateResource(Type** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new Type(env_);
    return Status::OK();
  }
  mutex mu_;
  Env* env_;
};

template<typename Type>
class IOIndexableGetItemOp : public OpKernel {
 public:
  explicit IOIndexableGetItemOp<Type>(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* context) override {
    Type* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    int64 start = start_tensor->scalar<int64>()();

    const Tensor* stop_tensor;
    OP_REQUIRES_OK(context, context->input("stop", &stop_tensor));
    int64 stop = stop_tensor->scalar<int64>()();

    const Tensor* step_tensor;
    OP_REQUIRES_OK(context, context->input("step", &step_tensor));
    int64 step = step_tensor->scalar<int64>()();

    OP_REQUIRES(context, (step == 1), errors::InvalidArgument("step != 1 is not supported: ", step));

    std::vector<DataType> dtypes;
    std::vector<PartialTensorShape> shapes;
    OP_REQUIRES_OK(context, resource->Spec(dtypes, shapes));

    int64 count = shapes[0].dim_size(0);
    if (start > count) {
      start = count;
    }
    if (stop < 0) {
      stop = count;
    }
    if (stop < start) {
      stop = start;
    }

    std::vector<Tensor> tensors;
    for (size_t i = 0; i < dtypes.size(); i++) {
      gtl::InlinedVector<int64, 4> dims = shapes[i].dim_sizes();
      dims[0] = stop - start;
      tensors.emplace_back(Tensor(dtypes[i], TensorShape(dims)));
    }
    OP_REQUIRES_OK(context, resource->GetItem(start, stop, step, tensors));
    for (size_t i = 0; i < tensors.size(); i++) {
      context->set_output(i, tensors[i]);
    }
  }
};
}  // namespace data
}  // namespace tensorflow
