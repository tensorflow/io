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
  virtual Status Output() override {
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
  Status Initialize(const std::vector<string>& destination) {
    destination_ = destination;
    if (destination_.size() != 1) {
      return errors::Unimplemented("only one file is supported: ", destination_.size());
    }
    return Status::OK();
  }
 private:
  std::vector<string> destination_ GUARDED_BY(mu_);
};

class TextOutputSequenceOp : public OutputSequenceOp<TextOutputSequence> {
 public:
  explicit TextOutputSequenceOp(OpKernelConstruction* context)
    : OutputSequenceOp<TextOutputSequence>(context) {
  }
 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<TextOutputSequence>::Compute(context);
    mutex_lock l(mu_);
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
};

REGISTER_KERNEL_BUILDER(Name("TextOutputSequence").Device(DEVICE_CPU),
                        TextOutputSequenceOp);


REGISTER_KERNEL_BUILDER(Name("TextOutputSequenceSetItem").Device(DEVICE_CPU),
                        OutputSequenceSetItemOp<TextOutputSequence>);
}  // namespace tensorflow
