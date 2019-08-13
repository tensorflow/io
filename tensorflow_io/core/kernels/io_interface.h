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
virtual ~IOInterface() {}
  virtual Status Init(const string& input, const void* memory_data, const int64 memory_size, const string& metadata, DataType* dtype, PartialTensorShape* shape) = 0;
};

class IOIterableInterface : public IOInterface {
 public:
  virtual Status Next(const int64 record_to_read, Tensor* value, Tensor* key) = 0;
};

class IOIndexableInterface : public IOInterface {
 public:
virtual ~IOIndexableInterface() {}
  virtual Status GetItem(const int64 start, const int64 stop, const int64 step, Tensor *value, Tensor* key) = 0;
  virtual Status Len(int64 *len) = 0;
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
    const Tensor& input_tensor = context->input(0);
    const string& input = input_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(1);
    const string& memory = memory_tensor.scalar<string>()();

    const Tensor& metadata_tensor = context->input(2);
    const string& metadata = metadata_tensor.scalar<string>()();

    DataType dtype;
    PartialTensorShape shape;
    OP_REQUIRES_OK(context, this->resource_->Init(input, memory.data(), memory.size(), metadata, &dtype, &shape));
    OP_REQUIRES(context, (shape.dim_size(0) < 0), errors::InvalidArgument("shape[0] should be None, received ", shape));
    for (int64 i = 1; i < shape.dims(); i++) {
      OP_REQUIRES(context, (shape.dim_size(i) > 0), errors::InvalidArgument("shape[", i, "] should not be None, received ", shape));
    }
    Tensor dtype_tensor(DT_INT64, TensorShape({}));
    dtype_tensor.scalar<int64>()() = dtype;
    context->set_output(1, dtype_tensor);
    Tensor shape_tensor(DT_INT64, TensorShape({shape.dims()}));
    for (int64 i = 0; i < shape.dims(); i++) {
      shape_tensor.flat<int64>()(i) = shape.dim_size(i);
    }
    context->set_output(2, shape_tensor);
  }
  Status CreateResource(Type** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new Type(env_);
    return Status::OK();
  }
  Env* env_;
};
template<typename Type>
class IOIndexableGetItemOp : public OpKernel {
 public:
  explicit IOIndexableGetItemOp<Type>(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& start_tensor = context->input(1);
    int64 start = start_tensor.scalar<int64>()();

    const Tensor& stop_tensor = context->input(2);
    int64 stop = stop_tensor.scalar<int64>()();

    const Tensor& step_tensor = context->input(3);
    int64 step = step_tensor.scalar<int64>()();

    OP_REQUIRES(context, (step == 1), errors::InvalidArgument("step != 1 is not supported: ", step));

    Type* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "input", &resource));
    int64 count = 0;
    Status status = resource->Len(&count);
    if (!status.ok()) {
      resource->Unref();
      OP_REQUIRES_OK(context, status);
    }
    if (start > count) {
      start = count;
    }
    if (stop < 0) {
      stop = count;
    }
    if (stop < start) {
      stop = start;
    }

    Tensor value, key;
    status = resource->GetItem(start, stop, step, &value, &key);
    resource->Unref();
    OP_REQUIRES_OK(context, status);
    context->set_output(0, value);
    context->set_output(1, key);
  }
};
template<typename Type>
class IOIndexableLenOp : public OpKernel {
 public:
  explicit IOIndexableLenOp<Type>(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* context) override {
    Type* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "input", &resource));
    int64 count = 0;
    Status status = resource->Len(&count);
    resource->Unref();
    OP_REQUIRES_OK(context, status);
    Tensor len_tensor(DT_INT64, TensorShape({}));
    len_tensor.scalar<int64>()() = count;
    context->set_output(0, len_tensor);
  }
};
template<typename Type>
class IOIterableNextOp : public OpKernel {
 public:
  explicit IOIterableNextOp<Type>(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* context) override {
    Type* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "input", &resource));

    const Tensor& capacity_tensor = context->input(1);
    const int64 capacity = capacity_tensor.scalar<int64>()();

    Tensor value, key;
    int64 record_to_read = capacity;

    Status status = resource->Next(record_to_read, &value, &key);
    resource->Unref();
    OP_REQUIRES_OK(context, status);

    context->set_output(0, value);
    context->set_output(1, key);
  }
};
}  // namespace data
}  // namespace tensorflow
