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
  virtual Status Spec(const Tensor& component, PartialTensorShape* shape, DataType* dtype, bool label) = 0;

  virtual Status Components(Tensor* components) {
    // By default there is only one component: Unimplemented
    return errors::Unimplemented("Component");
  }
  virtual Status Extra(const Tensor& component, std::vector<Tensor>* extra) {
    // This is the chance to provide additional extra information which should be appended to extra.
    return errors::Unimplemented("Extra");
  }
  virtual Status Context(OpKernelContext* context) {
    // This is the time to attach another resource to this interface.
    return errors::Unimplemented("Context");
  }
};

class IOIterableInterface : public IOInterface {
 public:
  // Check value==nullptr or label==nullptr to see which field is needed.
  virtual Status Next(const int64 capacity, const Tensor& component, int64* record_read, Tensor* value, Tensor* label) = 0;
};

class IOIndexableInterface : public IOInterface {
 public:
  // Check value==nullptr or label==nullptr to see which field is needed.
  virtual Status Read(const int64 start, const int64 stop, const Tensor& component, Tensor* value, Tensor* label) = 0;
};

class IOMappingInterface : public IOInterface {
 public:
  virtual Status Read(const Tensor& key, Tensor* value) = 0;
};


template<typename Type>
class IOIndexableImplementation : public IOIndexableInterface {
 public:
  IOIndexableImplementation<Type>(Env* env)
  : env_(env)
  , iterable_(nullptr) {}

  ~IOIndexableImplementation<Type>() {
    if (iterable_) {
      iterable_->Unref();
    }
  }

  virtual Status Context(OpKernelContext* context) {
    return GetResourceFromContext(context, "iterable", &iterable_);
  }

  Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) override {
    TF_RETURN_IF_ERROR(iterable_->Init(input, metadata, memory_data, memory_size));
    // We assume only one component at the moment, we also assume only value field is needed
    Tensor component(DT_INT64, TensorShape({}));
    component.scalar<int64>()() = 0;

    const int64 capacity = 4096;

    TF_RETURN_IF_ERROR(iterable_->Spec(component, &value_shape_, &value_dtype_, false));
    gtl::InlinedVector<int64, 4> dims = value_shape_.dim_sizes();
    dims[0] = capacity;
    TensorShape chunk_value_shape(dims);

    int64 total = 0;

    int64 record_read = 0;
    do {
      chunk_values_.push_back(Tensor(value_dtype_, chunk_value_shape));
      TF_RETURN_IF_ERROR(iterable_->Next(capacity, component, &record_read, &chunk_values_.back(), nullptr));
      if (record_read == 0) {
        chunk_values_.pop_back();
        break;
      }
      if (record_read < capacity) {
        chunk_values_.back() = chunk_values_.back().Slice(0, record_read);
      }
      total += record_read;
    } while (record_read != 0);
    value_shape_.set_dim(0, total);
    return Status::OK();
  }
  virtual Status Spec(const Tensor& component, PartialTensorShape* shape, DataType* dtype, bool label) override {
    // We assume only one component at the moment, we also assume only value field is needed
    if (component.scalar<int64>()() != 0) {
        return errors::InvalidArgument("component ", component.scalar<int64>()(), " not supported");
    }
    if (label) {
        return errors::InvalidArgument("label is not supported");
    }
    *shape = value_shape_;
    *dtype = value_dtype_;
    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("IOIndexableImplementation<", iterable_->DebugString(), ">[]");
  }

  Status Read(const int64 start, const int64 stop, const Tensor& component, Tensor* value, Tensor* label) override {
    return GetTensor(value_shape_, value_dtype_, chunk_values_, start, stop, component, value);
  }

 private:
  Status GetTensor(const PartialTensorShape& shape, const DataType& dtype, const std::vector<Tensor>& chunk_tensors, const int64 start, const int64 stop, const Tensor& component, Tensor* tensor) {
    // We assume only one component at the moment.
    if (component.scalar<int64>()() != 0) {
        return errors::InvalidArgument("component ", component.scalar<int64>()(), " not supported");
    }
    // Find first chunk
    int64 chunk_index = 0;
    int64 chunk_element = -1;
    int64 current_element = 0;
    while (chunk_index < chunk_tensors.size()) {
      if (current_element <= start && start < current_element + chunk_tensors[chunk_index].shape().dim_size(0)) {
        chunk_element = start - current_element;
        current_element = start;
        break;
      }
      current_element += chunk_tensors[chunk_index].shape().dim_size(0);
      chunk_index++;
    }
    if (chunk_element < 0) {
      return errors::InvalidArgument("start is out of range: ", start);
    }
    TensorShape element_shape(shape.dim_sizes());
    element_shape.RemoveDim(0);
    Tensor element(dtype, element_shape);

    while (current_element < stop) {
      batch_util::CopySliceToElement(chunk_tensors[chunk_index], &element, chunk_element);
      batch_util::CopyElementToSlice(element, tensor, (current_element - start));
      chunk_element++;
      if (chunk_element == chunk_tensors[chunk_index].shape().dim_size(0)) {
        chunk_index++;
        chunk_element = 0;
      }
      current_element++;
    }
    return Status::OK();
  }

  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  Type* iterable_ GUARDED_BY(mu_);
  PartialTensorShape value_shape_ GUARDED_BY(mu_);
  DataType value_dtype_ GUARDED_BY(mu_);
  std::vector<Tensor> chunk_values_;
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

    Status status = this->resource_->Context(context);
    if (!errors::IsUnimplemented(status)) {
      OP_REQUIRES_OK(context, status);
    }

    std::vector<string> input;
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    for (int64 i = 0; i < input_tensor->NumElements(); i++) {
        input.push_back(input_tensor->flat<string>()(i));
    }

    std::vector<string> metadata;
    const Tensor* metadata_tensor;
    status = context->input("metadata", &metadata_tensor);
    if (status.ok()) {
      for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
        metadata.push_back(metadata_tensor->flat<string>()(i));
      }
    }

    size_t memory_size = 0;
    const void *memory_data = nullptr;
    const Tensor* memory_tensor;
    status = context->input("memory", &memory_tensor);
    if (status.ok()) {
      memory_data = memory_tensor->scalar<string>()().data();
      memory_size = memory_tensor->scalar<string>()().size();
    }

    OP_REQUIRES_OK(context, this->resource_->Init(input, metadata, memory_data, memory_size));
    Tensor components_tensor;
    status = this->resource_->Components(&components_tensor);
    if (!errors::IsUnimplemented(status)) {
      OP_REQUIRES_OK(context, status);
      context->set_output(1, components_tensor);
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
    OP_REQUIRES_OK(context, resource->Spec(*component, &shape, &dtype, false));

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
      : OpKernel(ctx), value_output_(true), label_output_(false) {
    std::vector<string> filter;
    Status status = ctx->GetAttr("filter", &filter);
    if (status.ok()) {
      if (filter.size() != 0) {
        value_output_ = false;
        label_output_ = false;
        for (size_t i = 0; i < filter.size(); i++) {
          if (filter[i] == "value") {
            value_output_ = true;
          }
          if (filter[i] == "label") {
            label_output_ = true;
          }
        }
      }
    }
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

    Tensor* value_tensor = nullptr;
    Tensor value;
    if (value_output_) {
      PartialTensorShape shape;
      DataType dtype;
      OP_REQUIRES_OK(context, resource->Spec(*component, &shape, &dtype, false));
      gtl::InlinedVector<int64, 4> dims = shape.dim_sizes();
      dims[0] = capacity;
      TensorShape value_shape(dims);
      value = Tensor(dtype, value_shape);
      value_tensor = &value;
    }

    Tensor* label_tensor = nullptr;
    Tensor label;
    if (label_output_) {
      PartialTensorShape shape;
      DataType dtype;
      OP_REQUIRES_OK(context, resource->Spec(*component, &shape, &dtype, true));
      gtl::InlinedVector<int64, 4> dims = shape.dim_sizes();
      dims[0] = capacity;
      TensorShape label_shape(dims);
      label = Tensor(dtype, label_shape);
      label_tensor = &label;
    }

    int64 record_read;
    OP_REQUIRES_OK(context, resource->Next(capacity, *component, &record_read, value_tensor, label_tensor));

    int64 output_index = 0;
    if (record_read < capacity) {
      if (value_output_) {
        context->set_output(output_index, value.Slice(0, record_read));
        output_index++;
      }
      if (label_output_) {
        context->set_output(output_index, label.Slice(0, record_read));
        output_index++;
      }
    } else {
      if (value_output_) {
        context->set_output(output_index, value);
        output_index++;
      }
      if (label_output_) {
        context->set_output(output_index, label);
        output_index++;
      }
    }
  }
 private:
  bool value_output_;
  bool label_output_;
};
template<typename Type>
class IOIndexableReadOp : public OpKernel {
 public:
  explicit IOIndexableReadOp<Type>(OpKernelConstruction* ctx)
      : OpKernel(ctx) , value_output_(true), label_output_(false) {
    std::vector<string> filter;
    Status status = ctx->GetAttr("filter", &filter);
    if (status.ok()) {
      if (filter.size() != 0) {
        value_output_ = false;
        label_output_ = false;
        for (size_t i = 0; i < filter.size(); i++) {
          if (filter[i] == "value") {
            value_output_ = true;
          }
          if (filter[i] == "label") {
            label_output_ = true;
          }
        }
      }
    }
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

    const Tensor* component;
    OP_REQUIRES_OK(context, context->input("component", &component));

    int64 output_index = 0;
    Tensor* value_tensor = nullptr;
    if (value_output_) {
      PartialTensorShape shape;
      DataType dtype;
      OP_REQUIRES_OK(context, resource->Spec(*component, &shape, &dtype, false));
      gtl::InlinedVector<int64, 4> dims = shape.dim_sizes();
      dims[0] = stop - start;
      TensorShape value_shape(dims);
      OP_REQUIRES_OK(context, context->allocate_output(output_index, value_shape, &value_tensor));
      output_index++;
    }
    Tensor* label_tensor = nullptr;
    if (label_output_) {
      PartialTensorShape shape;
      DataType dtype;
      OP_REQUIRES_OK(context, resource->Spec(*component, &shape, &dtype, true));
      gtl::InlinedVector<int64, 4> dims = shape.dim_sizes();
      dims[0] = stop - start;
      TensorShape label_shape(dims);
      OP_REQUIRES_OK(context, context->allocate_output(output_index, label_shape, &label_tensor));
      output_index++;
    }
    OP_REQUIRES_OK(context, resource->Read(start, stop, *component, value_tensor, label_tensor));
  }
 private:
  bool value_output_;
  bool label_output_;
};
template<typename Type>
class IOMappingReadOp : public OpKernel {
 public:
  explicit IOMappingReadOp<Type>(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* context) override {
    Type* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* key;
    OP_REQUIRES_OK(context, context->input("key", &key));

    Tensor value(DT_STRING, TensorShape({key->NumElements()}));
    OP_REQUIRES_OK(context, resource->Read(*key, &value));
    context->set_output(0, value);
  }
};
}  // namespace data
}  // namespace tensorflow
