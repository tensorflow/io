/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/public/version.h"
#include "kernels/sequence_ops.h"

#include <deque>

namespace tensorflow {
class TextOutputSequence : public OutputSequence {
 public:
  TextOutputSequence(Env* env)
   : OutputSequence(env) {}

  virtual ~TextOutputSequence() override {}
  virtual Status SetItem(int64 index, const void *item) {
    if (destination_.size() != 1) {
      return errors::Unimplemented("only one file is supported: ", destination_.size());
    }
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
      std::unique_ptr<WritableFile> file;
      TF_RETURN_IF_ERROR(env_->NewAppendableFile(destination_[0], &file));
      while (fifo_.size() != 0 && fifo_.front().get() != nullptr) {
	TF_RETURN_IF_ERROR(file->Append(strings::StrCat(fifo_.front().get()->c_str(), "\n")));
        fifo_.pop_front();
        base_++;
      }
      TF_RETURN_IF_ERROR(file->Close());
    }
    return Status::OK();
  }
#if TF_MAJOR_VERSION==1&&TF_MINOR_VERSION==13
  virtual string DebugString() {
#else
  virtual string DebugString() const {
#endif
    return strings::StrCat("TextOutputSequence[]");
  }
 private:
  int64 base_ GUARDED_BY(mu_) = 0;
  std::deque<std::unique_ptr<string>> fifo_ GUARDED_BY(mu_);
};

class TextOutputSequenceOp : public ResourceOpKernel<TextOutputSequence> {
 public:
  explicit TextOutputSequenceOp(OpKernelConstruction* context)
    : ResourceOpKernel<TextOutputSequence>(context) {
    env_ = context->env();
  }
 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<TextOutputSequence>::Compute(context);
    const Tensor* destination_tensor;
    OP_REQUIRES_OK(context, context->input("destination", &destination_tensor));
    OP_REQUIRES(
        context, destination_tensor->dims() <= 1,
        errors::InvalidArgument("`destination` must be a scalar or vector."));

    std::vector<string> destination;
    destination.reserve(destination_tensor->NumElements());
    for (int i = 0; i < destination_tensor->NumElements(); ++i) {
      destination.push_back(destination_tensor->flat<string>()(i));
    }

    OP_REQUIRES_OK(context, resource_->Initialize(destination));
  }
  Status CreateResource(TextOutputSequence** sequence)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *sequence = new TextOutputSequence(env_);
    return Status::OK();
  }
  Env* env_;
};

class TextOutputSequenceSetItemOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  void Compute(OpKernelContext* ctx) override {
    TextOutputSequence* sequence;
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
};


REGISTER_KERNEL_BUILDER(Name("TextOutputSequence").Device(DEVICE_CPU),
                        TextOutputSequenceOp);


REGISTER_KERNEL_BUILDER(Name("TextOutputSequenceSetItem").Device(DEVICE_CPU),
                        TextOutputSequenceSetItemOp);
}  // namespace tensorflow
