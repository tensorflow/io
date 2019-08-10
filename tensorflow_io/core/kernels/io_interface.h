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

class IOIterableInterface : public ResourceBase {
 public:
  virtual Status Init(const string& input, const void* memory_data, const int64 memory_size, const string& metadata) = 0;
  virtual Status Next(const int64 record_to_read, std::vector<Tensor>& tensors, int64* record_read) = 0;

  virtual Status GetShape(std::vector<TensorShape>& shapes) = 0;
};

class IOIndexableInterface : public IOIterableInterface {
 public:
  virtual Status GetItem(const int64 start, const int64 stop, std::vector<Tensor>& tensors) = 0;
  virtual Status Len(int64 *len) = 0;

 protected:
  virtual ~IOIndexableInterface() {}
 private:
  Status Next(const int64 record_to_read, std::vector<Tensor>& tensors, int64* record_read) override {
    int64 end;
    TF_RETURN_IF_ERROR(Len(&end));
    int64 start = record_offset_;
    if (start > end) {
      start = end;
    }
    int64 stop = start + record_to_read;
    if (stop > end) {
      stop = end;
    }
    TF_RETURN_IF_ERROR(GetItem(start, stop, tensors));
    (*record_read) = stop - start;
    record_offset_ += (*record_read);
  }
  int64 record_offset_ = 0;
  
};

template<typename Type>
class IOIndexableReadOp : public OpKernel {
 public:
  explicit IOIndexableReadOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    env_ = ctx->env();
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const string& input = input_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(1);
    const string& memory = memory_tensor.scalar<string>()();

    const Tensor& metadata_tensor = context->input(2);
    const string& metadata = metadata_tensor.scalar<string>()();

    const Tensor& start_tensor = context->input(3);
    int64 start = start_tensor.scalar<int64>()();

    const Tensor& stop_tensor = context->input(4);
    int64 stop = stop_tensor.scalar<int64>()();

    std::unique_ptr<Type> resource = std::unique_ptr<Type>(new Type(env_));
    OP_REQUIRES_OK(context, resource->Init(input, memory.data(), memory.size(), metadata));

    int64 total = 0;
    OP_REQUIRES_OK(context, resource->Len(&total));
    if (start > total) {
      start = total;
    }
    if (stop < 0) {
      stop = total;
    }
    if (stop < start) {
      stop = start;
    }

    std::vector<TensorShape> shapes;
    OP_REQUIRES_OK(context, resource->GetShape(shapes));

    std::vector<Tensor> tensors;
    for (int64 i = 0; i < dtypes_.size(); i++) {
      TensorShape shape = shapes[i];
      shape.InsertDim(0, stop - start);

      Tensor tensor(dtypes_[i], shape);

      tensors.emplace_back(tensor);
    }
    OP_REQUIRES_OK(context, resource->GetItem(start, stop, tensors));
    for (int64 i = 0; i < tensors.size(); i++) {
      context->set_output(i, tensors[i]);
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::vector<DataType> dtypes_;
};

template<typename Type>
class IOIterableInitOp : public ResourceOpKernel<IOIterableInterface> {
 public:
  explicit IOIterableInitOp<Type>(OpKernelConstruction* context)
      : ResourceOpKernel<IOIterableInterface>(context) {
    env_ = context->env();
  }
 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<IOIterableInterface>::Compute(context);
    const Tensor& input_tensor = context->input(0);
    const string& input = input_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(1);
    const string& memory = memory_tensor.scalar<string>()();

    const Tensor& metadata_tensor = context->input(2);
    const string& metadata = metadata_tensor.scalar<string>()();
    OP_REQUIRES_OK(context, resource_->Init(input, memory.data(), memory.size(), metadata));
  }
  Status CreateResource(IOIterableInterface** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new Type(env_);
    return Status::OK();
  }
  Env* env_;
};
template<typename Type>
class IOIterableNextOp : public OpKernel {
 public:
  explicit IOIterableNextOp<Type>(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
  }

  void Compute(OpKernelContext* context) override {
    IOIterableInterface* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "input", &resource));

    const Tensor& capacity_tensor = context->input(1);
    const int64 capacity = capacity_tensor.scalar<int64>()();

    std::vector<TensorShape> shapes;
    OP_REQUIRES_OK(context, resource->GetShape(shapes));

    std::vector<Tensor> tensors;
    for (int64 i = 0; i < dtypes_.size(); i++) {
      TensorShape shape = shapes[i];
      shape.InsertDim(0, capacity);

      Tensor tensor(dtypes_[i], shape);

      tensors.emplace_back(tensor);
    }

    int64 record_to_read = capacity;

    int64 record_read = 0;
    Status status = resource->Next(record_to_read, tensors, &record_read);
    resource->Unref();
    OP_REQUIRES_OK(context, status);
    for (int64 i = 0; i < tensors.size(); i++) {
      if (record_read < record_to_read) {
        // In case record_read is smaller than record_to_read, cut a slice
        Tensor output_tensor = tensors[i].Slice(0, record_read);
        context->set_output(i, output_tensor);
      } else {
        context->set_output(i, tensors[i]);
      }
    }
  }
private:
  std::vector<DataType> dtypes_;
};
template<typename Type>
class IOIterableReadOp : public OpKernel {
 public:
  explicit IOIterableReadOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    env_ = ctx->env();
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const string& input = input_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(1);
    const string& memory = memory_tensor.scalar<string>()();

    const Tensor& metadata_tensor = context->input(2);
    const string& metadata = metadata_tensor.scalar<string>()();

    std::unique_ptr<Type> resource = std::unique_ptr<Type>(new Type(env_));
    OP_REQUIRES_OK(context, resource->Init(input, memory.data(), memory.size(), metadata));

    int64 capacity = 65536; // TODO: obtain capacity from parameters?

    std::vector<TensorShape> shapes;
    OP_REQUIRES_OK(context, resource->GetShape(shapes));
    std::vector<std::vector<Tensor>> tensors_vector;
    std::vector<int64> tensors_record_read;
    int64 total_read = 0;
    do {
      tensors_vector.emplace_back(std::vector<Tensor>());
      for (int64 i = 0; i < dtypes_.size(); i++) {
        TensorShape shape = shapes[i];
        shape.InsertDim(0, capacity);

        Tensor tensor(dtypes_[i], shape);

        tensors_vector.back().emplace_back(tensor);
      }
      int64 record_to_read = capacity;

      int64 record_read = 0;
      OP_REQUIRES_OK(context, resource->Next(record_to_read, tensors_vector.back(), &record_read));

      total_read += record_read;

      if (record_read == 0) {
        break;
      }
      tensors_record_read.emplace_back(record_read);
    } while (true);

    for (int64 i = 0; i < dtypes_.size(); i++) {
      TensorShape shape = shapes[i];

      Tensor element(dtypes_[i], shape);

      shape.InsertDim(0, total_read);

      Tensor tensor(dtypes_[i], shape);

      // Copy tensors_vector, distribute copy?
      int64 slice_index = 0;
      for (size_t tensors_record_index = 0; tensors_record_index < tensors_record_read.size(); tensors_record_index++) {
        for (int64 element_index = 0; slice_index < total_read && element_index < tensors_record_read[tensors_record_index]; element_index++) {
          OP_REQUIRES_OK(context, batch_util::CopySliceToElement(tensors_vector[tensors_record_index][i], &element, element_index));
          OP_REQUIRES_OK(context, batch_util::CopyElementToSlice(element, &tensor, slice_index));
          slice_index++;
        }
      }
      context->set_output(i, tensor);
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::vector<DataType> dtypes_;
};

}  // namespace data
}  // namespace tensorflow
