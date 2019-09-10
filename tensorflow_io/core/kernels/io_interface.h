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
  virtual Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) = 0;
  virtual Status Spec(const Tensor& component, PartialTensorShape* shape, DataType* dtype) = 0;

  virtual Status Component(Tensor* component) {
    // By default there is only one component: Unimplemented
    return errors::Unimplemented("Component");
  }
  virtual Status Extra(const Tensor& component, std::vector<Tensor>* extra) {
    // This is the chance to provide additional extra information which should be appended to extra.
    return errors::Unimplemented("Extra");
  }
};

class IOIterableInterface : public IOInterface {
 public:
  virtual Status Next(const int64 capacity, const Tensor& component, Tensor* tensor, int64* record_read) = 0;
};

class IOIndexableInterface : public IOInterface {
 public:
  virtual Status GetItem(const int64 start, const int64 stop, const int64 step, const Tensor& component, Tensor* tensor) = 0;
};

class IOMappingInterface : public IOInterface {
 public:
  virtual Status GetItem(const Tensor& key, Tensor* tensor) = 0;
};

template<typename Type>
class IOIndexableImplementation : public IOIndexableInterface {
 public:
  IOIndexableImplementation<Type>(Env* env)
  : env_(env)
  , iterable_(new Type(env)) {}

  ~IOIndexableImplementation<Type>() {}
  Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) override {
    TF_RETURN_IF_ERROR(iterable_->Init(input, metadata, memory_data, memory_size));
    TF_RETURN_IF_ERROR(iterable_->Spec(shapes_, dtypes_));

    const int64 capacity = 4096;
    std::vector<TensorShape> chunk_shapes;
    for (size_t component = 0; component < shapes_.size(); component++) {
      gtl::InlinedVector<int64, 4> dims = shapes_[component].dim_sizes();
      dims[0] = capacity;
      chunk_shapes.push_back(TensorShape(dims));
    }

    int64 total = 0;

    int64 record_read = 0;
    do {
      chunk_tensors_.push_back(std::vector<Tensor>());
      for (size_t component = 0; component < shapes_.size(); component++) {
        chunk_tensors_.back().push_back(Tensor(dtypes_[component], chunk_shapes[component]));
        int64 chunk_record_read = 0;
        TF_RETURN_IF_ERROR(iterable_->Next(capacity, component, &chunk_tensors_.back()[component], &chunk_record_read));
        if (component != 0 && record_read != chunk_record_read) {
          return errors::Internal("component ", component, " has differtnt chunk size: ", chunk_record_read, " vs. ", record_read);
        }
        record_read = chunk_record_read;
      }
      if (record_read == 0) {
        chunk_tensors_.pop_back();
        break;
      }
      if (record_read < capacity) {
        for (size_t component = 0; component < shapes_.size(); component++) {
          chunk_tensors_.back()[component] = chunk_tensors_.back()[component].Slice(0, record_read);
        }
      }
      total += record_read;
    } while (record_read != 0);
    for (size_t component = 0; component < shapes_.size(); component++) {
      shapes_[component].set_dim(0, total);
    }
    return Status::OK();
  }
  virtual Status Spec(std::vector<PartialTensorShape>& shapes, std::vector<DataType>& dtypes) override {
    for (size_t component = 0; component < shapes_.size(); component++) {
      shapes.push_back(shapes_[component]);
    }
    for (size_t component = 0; component < dtypes_.size(); component++) {
      dtypes.push_back(dtypes_[component]);
    }
    return Status::OK();
  }

  Status Extra(std::vector<Tensor>* extra) override {
    return iterable_->Extra(extra);
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("IOIndexableImplementation<", iterable_->DebugString(), ">[]");
  }

  Status GetItem(const int64 start, const int64 stop, const int64 step, const int64 component, Tensor* tensor) override {
    if (step != 1) {
      return errors::InvalidArgument("step != 1 is not supported: ", step);
    }
    // Find first chunk
    int64 chunk_index = 0;
    int64 chunk_element = -1;
    int64 current_element = 0;
    while (chunk_index < chunk_tensors_.size()) {
      if (current_element <= start && start < current_element + chunk_tensors_[chunk_index][component].shape().dim_size(0)) {
        chunk_element = start - current_element;
        current_element = start;
        break;
      }
      current_element += chunk_tensors_[chunk_index][component].shape().dim_size(0);
      chunk_index++;
    }
    if (chunk_element < 0) {
      return errors::InvalidArgument("start is out of range: ", start);
    }
    TensorShape shape(shapes_[component].dim_sizes());
    shape.RemoveDim(0);
    Tensor element(dtypes_[component], shape);

    while (current_element < stop) {
      batch_util::CopySliceToElement(chunk_tensors_[chunk_index][component], &element, chunk_element);
      batch_util::CopyElementToSlice(element, tensor, (current_element - start));
      chunk_element++;
      if (chunk_element == chunk_tensors_[chunk_index][component].shape().dim_size(0)) {
        chunk_index++;
        chunk_element = 0;
      }
      current_element++;
    }
    return Status::OK();
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<Type> iterable_ GUARDED_BY(mu_);
  std::vector<PartialTensorShape> shapes_ GUARDED_BY(mu_);
  std::vector<DataType> dtypes_ GUARDED_BY(mu_);
  std::vector<std::vector<Tensor>> chunk_tensors_;
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
    std::vector<string> input;
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    for (int64 i = 0; i < input_tensor->NumElements(); i++) {
        input.push_back(input_tensor->flat<string>()(i));
    }

    Status status;

    std::vector<string> metadata;
    const Tensor* metadata_tensor;
    status = context->input("metadata", &metadata_tensor);
    if (status.ok()) {
      for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
        metadata.push_back(metadata_tensor->flat<string>()(i));
      }
    }

    const void *memory_data = nullptr;
    size_t memory_size = 0;

    const Tensor* memory_tensor;
    status = context->input("memory", &memory_tensor);
    if (status.ok()) {
      memory_data = memory_tensor->scalar<string>()().data();
      memory_size = memory_tensor->scalar<string>()().size();
    }

    OP_REQUIRES_OK(context, this->resource_->Init(input, metadata, memory_data, memory_size));
    Tensor component_tensor;
    status = this->resource_->Component(&component_tensor);
    if (!errors::IsUnimplemented(status)) {
      OP_REQUIRES_OK(context, status);
      context->set_output(1, component_tensor);
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
class IOInterfaceSpecOp : public OpKernel {
 public:
  explicit IOInterfaceSpecOp<Type>(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* context) override {
    Type* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* component;
    OP_REQUIRES_OK(context, context->input("component", &component));

    PartialTensorShape shape;
    DataType dtype;
    OP_REQUIRES_OK(context, resource->Spec(*component, &shape, &dtype));

    Tensor shape_tensor(DT_INT64, TensorShape({shape.dims()}));
    for (int64 i = 0; i < shape.dims(); i++) {
      shape_tensor.flat<int64>()(i) = shape.dim_size(i);
    }
    Tensor dtype_tensor(DT_INT64, TensorShape({}));
    dtype_tensor.scalar<int64>()() = dtype;
    context->set_output(0, shape_tensor);
    context->set_output(1, dtype_tensor);

    std::vector<Tensor> extra;
    Status status = resource->Extra(*component, &extra);
    if (!errors::IsUnimplemented(status)) {
      OP_REQUIRES_OK(context, status);
      for (size_t i = 0; i < extra.size(); i++) {
        context->set_output(2 + i, extra[i]);
      }
    }
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
    core::ScopedUnref unref(resource);

    const Tensor* capacity_tensor;
    OP_REQUIRES_OK(context, context->input("capacity", &capacity_tensor));
    const int64 capacity = capacity_tensor->scalar<int64>()();

    const Tensor* component;
    OP_REQUIRES_OK(context, context->input("component", &component));

    OP_REQUIRES(context, (capacity > 0), errors::InvalidArgument("capacity <= 0 is not supported: ", capacity));

    PartialTensorShape shape;
    DataType dtype;
    OP_REQUIRES_OK(context, resource->Spec(*component, &shape, &dtype));

    gtl::InlinedVector<int64, 4> dims = shape.dim_sizes();
    dims[0] = capacity;
    Tensor tensor(dtype, TensorShape(dims));

    int64 record_read;
    OP_REQUIRES_OK(context, resource->Next(capacity, *component, &tensor, &record_read));
    if (record_read < capacity) {
      context->set_output(0, tensor.Slice(0, record_read));
    } else {
      context->set_output(0, tensor);
    }
  }
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

    const Tensor* component;
    OP_REQUIRES_OK(context, context->input("component", &component));

    OP_REQUIRES(context, (step == 1), errors::InvalidArgument("step != 1 is not supported: ", step));

    PartialTensorShape shape;
    DataType dtype;
    OP_REQUIRES_OK(context, resource->Spec(*component, &shape, &dtype));

    int64 count = shape.dim_size(0);
    if (start > count) {
      start = count;
    }
    if (stop < 0) {
      stop = count;
    }
    if (stop < start) {
      stop = start;
    }

    gtl::InlinedVector<int64, 4> dims = shape.dim_sizes();
    dims[0] = stop - start;
    Tensor tensor(dtype, TensorShape(dims));
    OP_REQUIRES_OK(context, resource->GetItem(start, stop, step, *component, &tensor));
    context->set_output(0, tensor);
  }
};
template<typename Type>
class IOMappingGetItemOp : public OpKernel {
 public:
  explicit IOMappingGetItemOp<Type>(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* context) override {
    Type* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* key;
    OP_REQUIRES_OK(context, context->input("key", &key));

    Tensor tensor(DT_STRING, TensorShape({key->NumElements()}));
    OP_REQUIRES_OK(context, resource->GetItem(*key, &tensor));
    context->set_output(0, tensor);
  }
};
}  // namespace data
}  // namespace tensorflow
