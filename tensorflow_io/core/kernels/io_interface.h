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
  virtual Status Init(const std::vector<string>& input,
                      const std::vector<string>& metadata,
                      const void* memory_data, const int64 memory_size) = 0;
  virtual Status Spec(const string& component, PartialTensorShape* shape,
                      DataType* dtype, bool label) = 0;

  virtual Status Partitions(std::vector<int64>* partitions) {
    // By default partitions is not implemented: Unimplemented
    return errors::Unimplemented("Patitions");
  }
  virtual Status Components(std::vector<string>* components) {
    // By default there is only one component: Unimplemented
    return errors::Unimplemented("Components");
  }
  virtual Status Extra(const string& component, std::vector<Tensor>* extra) {
    // This is the chance to provide additional extra information which should
    // be appended to extra.
    return errors::Unimplemented("Extra");
  }
  virtual Status Context(OpKernelContext* context) {
    // This is the time to attach another resource to this interface.
    return errors::Unimplemented("Context");
  }
};

class IOReadableInterface : public IOInterface {
 public:
  // Check value==nullptr or label==nullptr to see which field is needed.
  virtual Status Read(const int64 start, const int64 stop,
                      const string& component, int64* record_read,
                      Tensor* value, Tensor* label) = 0;
};

class IOMappingInterface : public IOInterface {
 public:
  virtual Status Read(const Tensor& key, Tensor* value) = 0;
};

template <typename Type>
class IOInterfaceInitOp : public ResourceOpKernel<Type> {
 public:
  explicit IOInterfaceInitOp<Type>(OpKernelConstruction* context)
      : ResourceOpKernel<Type>(context) {
    env_ = context->env();
  }
  virtual ~IOInterfaceInitOp<Type>() {}

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
      input.push_back(input_tensor->flat<tstring>()(i));
    }

    std::vector<string> metadata;
    const Tensor* metadata_tensor;
    status = context->input("metadata", &metadata_tensor);
    if (status.ok()) {
      for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
        metadata.push_back(metadata_tensor->flat<tstring>()(i));
      }
    }

    size_t memory_size = 0;
    const void* memory_data = nullptr;
    const Tensor* memory_tensor;
    status = context->input("memory", &memory_tensor);
    if (status.ok()) {
      memory_data = memory_tensor->scalar<tstring>()().data();
      memory_size = memory_tensor->scalar<tstring>()().size();
    }

    OP_REQUIRES_OK(context, this->resource_->Init(input, metadata, memory_data,
                                                  memory_size));
    std::vector<string> components;
    status = this->resource_->Components(&components);
    if (!errors::IsUnimplemented(status)) {
      OP_REQUIRES_OK(context, status);
      Tensor components_tensor(
          DT_STRING, TensorShape({static_cast<int64>(components.size())}));
      for (size_t i = 0; i < components.size(); i++) {
        components_tensor.flat<tstring>()(i) = components[i];
      }
      context->set_output(1, components_tensor);
    }
  }
  Status CreateResource(Type** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new Type(env_);
    return Status::OK();
  }
  mutex mu_;
  Env* env_;
};

template <typename Type>
class IOInterfaceSpecOp : public OpKernel {
 public:
  explicit IOInterfaceSpecOp<Type>(OpKernelConstruction* ctx)
      : OpKernel(ctx), component_("") {
    string component;
    Status status = ctx->GetAttr("component", &component);
    if (status.ok()) {
      component_ = component;
    }
  }
  virtual ~IOInterfaceSpecOp<Type>() {}

  void Compute(OpKernelContext* context) override {
    Type* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    PartialTensorShape shape;
    DataType dtype;
    OP_REQUIRES_OK(context, resource->Spec(component_, &shape, &dtype, false));

    Tensor shape_tensor(DT_INT64, TensorShape({shape.dims()}));
    for (int64 i = 0; i < shape.dims(); i++) {
      shape_tensor.flat<int64>()(i) = shape.dim_size(i);
    }
    Tensor dtype_tensor(DT_INT64, TensorShape({}));
    dtype_tensor.scalar<int64>()() = dtype;
    context->set_output(0, shape_tensor);
    context->set_output(1, dtype_tensor);

    std::vector<Tensor> extra;
    Status status = resource->Extra(component_, &extra);
    if (!errors::IsUnimplemented(status)) {
      OP_REQUIRES_OK(context, status);
      for (size_t i = 0; i < extra.size(); i++) {
        context->set_output(2 + i, extra[i]);
      }
    }
  }

 private:
  string component_;
};

template <typename Type>
class IOReadableReadOp : public OpKernel {
 public:
  explicit IOReadableReadOp<Type>(OpKernelConstruction* ctx)
      : OpKernel(ctx),
        component_(""),
        value_output_(true),
        label_output_(false) {
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
    string component;
    status = ctx->GetAttr("component", &component);
    if (status.ok()) {
      component_ = component;
    }
  }
  virtual ~IOReadableReadOp<Type>() {}

  void Compute(OpKernelContext* context) override {
    Type* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    int64 start = start_tensor->scalar<int64>()();

    const Tensor* stop_tensor;
    OP_REQUIRES_OK(context, context->input("stop", &stop_tensor));
    int64 stop = stop_tensor->scalar<int64>()();

    Status status;

    Tensor* value_tensor = nullptr;
    Tensor value;
    if (value_output_) {
      PartialTensorShape shape;
      DataType dtype;
      OP_REQUIRES_OK(context,
                     resource->Spec(component_, &shape, &dtype, false));
      gtl::InlinedVector<int64, 4> dims = shape.dim_sizes();
      dims[0] = stop - start;
      TensorShape value_shape(dims);
      value = Tensor(dtype, value_shape);
      value_tensor = &value;
    }
    Tensor* label_tensor = nullptr;
    Tensor label;
    if (label_output_) {
      PartialTensorShape shape;
      DataType dtype;
      OP_REQUIRES_OK(context, resource->Spec(component_, &shape, &dtype, true));
      gtl::InlinedVector<int64, 4> dims = shape.dim_sizes();
      dims[0] = stop - start;
      TensorShape label_shape(dims);
      label = Tensor(dtype, label_shape);
      label_tensor = &label;
    }
    int64 record_read = 0;
    OP_REQUIRES_OK(context,
                   resource->Read(start, stop, component_, &record_read,
                                  value_tensor, label_tensor));
    int64 output_index = 0;
    if (record_read < stop - start) {
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
  string component_;
  bool value_output_;
  bool label_output_;
};
template <typename Type>
class IOMappingReadOp : public OpKernel {
 public:
  explicit IOMappingReadOp<Type>(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  virtual ~IOMappingReadOp<Type>() {}

  void Compute(OpKernelContext* context) override {
    Type* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* key;
    OP_REQUIRES_OK(context, context->input("key", &key));

    Tensor value(DT_STRING, TensorShape({key->NumElements()}));
    OP_REQUIRES_OK(context, resource->Read(*key, &value));
    context->set_output(0, value);
  }
};
template <typename Type>
class IOReadablePartitionsOp : public OpKernel {
 public:
  explicit IOReadablePartitionsOp<Type>(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  virtual ~IOReadablePartitionsOp<Type>() {}

  void Compute(OpKernelContext* context) override {
    Type* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    std::vector<int64> partitions;
    OP_REQUIRES_OK(context, resource->Partitions(&partitions));

    Tensor partitions_tensor(
        DT_INT64, TensorShape({static_cast<int64>(partitions.size())}));
    for (size_t i = 0; i < partitions.size(); i++) {
      partitions_tensor.flat<int64>()(i) = partitions[i];
    }

    context->set_output(0, partitions_tensor);
  }
};

}  // namespace data
}  // namespace tensorflow
