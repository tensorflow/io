/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <deque>

namespace tensorflow {
class OutputSequence : public ResourceBase {
 public:
  OutputSequence(Env* env)
   : env_(env) {}

  virtual ~OutputSequence() override {}
  virtual Status Flush() {
    return errors::Unimplemented("flush is not implemented");
  }
  virtual Status Output() = 0;
  virtual Status SetItem(int64 index, const void *item) {
    mutex_lock l(mu_);
    int64 size = fifo_.size();
    if (index < base_) {
      return errors::InvalidArgument("the item has already been add: ", index);
    }
    if (base_ <= index && index < base_ + size) {
      if (fifo_[index - base_].get() != nullptr) {
        return errors::InvalidArgument("the item has already been add before: ", index);
      }
      fifo_[index - base_].reset(new string((const char *)item));
    }
    if (base_ + size <= index) {
      for (int64 i = base_ + size; i < index; i++) {
        fifo_.push_back(nullptr);
      }
      fifo_.push_back(std::unique_ptr<string>(new string((const char *)item)));
    }
    if (fifo_.front().get() != nullptr) {
      TF_RETURN_IF_ERROR(Output());
    }
    return Status::OK();
  }
  virtual string DebugString() const {
    return strings::StrCat("OutputSequence[]");
  }
 protected:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  int64 base_ GUARDED_BY(mu_) = 0;
  std::deque<std::unique_ptr<string>> fifo_ GUARDED_BY(mu_);
};

template<typename T>
class OutputSequenceOp : public ResourceOpKernel<T> {
 public:
  explicit OutputSequenceOp<T>(OpKernelConstruction* context)
    : ResourceOpKernel<T>(context) {
    env_ = context->env();
  }
 protected:
  void Compute(OpKernelContext* context) = 0;
  Status CreateResource(T** sequence)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *sequence = new T(env_);
    return Status::OK();
  }
  Env* env_;
  mutex mu_;
};

template<typename T>
class OutputSequenceSetItemOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    T* sequence;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &sequence));
    core::ScopedUnref unref(sequence);
    const Tensor* index_tensor;
    const Tensor* item_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("index", &index_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(index_tensor->shape()),
                errors::InvalidArgument(
                    "Index tensor must be scalar, but had shape: ",
                    index_tensor->shape().DebugString()));
    OP_REQUIRES_OK(ctx, ctx->input("item", &item_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(item_tensor->shape()),
                errors::InvalidArgument(
                    "Item tensor must be scalar, but had shape: ",
                    item_tensor->shape().DebugString()));
    const int64 index = index_tensor->scalar<int64>()();
    const string& item = item_tensor->scalar<string>()();
    OP_REQUIRES_OK(ctx, sequence->SetItem(index, item.c_str()));
  }
 private:
  mutex mu_;
};

template<typename T>
class OutputSequenceFlushOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    T* sequence;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &sequence));
    core::ScopedUnref unref(sequence);
    OP_REQUIRES_OK(ctx, sequence->Flush());
  }
 private:
  mutex mu_;
};

}  // namespace tensorflow
