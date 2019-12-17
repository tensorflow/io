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
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {
namespace data {
namespace {

class LayerTextResource : public ResourceBase {
 public:
  LayerTextResource(Env* env) : env_(env) {}
  ~LayerTextResource() {
    if (file_.get() != nullptr) {
      file_->Close();
    }
  }

  Status Init(const string& filename) {
    TF_RETURN_IF_ERROR(env_->NewWritableFile(filename, &file_));
    return Status::OK();
  }
  Status Write(const Tensor& content) {
    mutex_lock l(mu_);
    for (int64 i = 0; i < content.NumElements(); i++) {
      TF_RETURN_IF_ERROR(file_->Append(content.flat<string>()(i)));
      TF_RETURN_IF_ERROR(file_->Append("\n"));
    }
    return Status::OK();
  }
  Status Sync() {
    TF_RETURN_IF_ERROR(file_->Flush());
    return Status::OK();
  }
  string DebugString() const override {
    return "LayerTextResource";
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<WritableFile> file_;
};

class LayerTextInitOp : public ResourceOpKernel<LayerTextResource> {
 public:
  explicit LayerTextInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<LayerTextResource>(context) {
    env_ = context->env();
  }
 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<LayerTextResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    OP_REQUIRES_OK(context, resource_->Init(input_tensor->scalar<string>()()));
  }
  Status CreateResource(LayerTextResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new LayerTextResource(env_);
    return Status::OK();
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};


class LayerTextCallOp : public OpKernel {
 public:
  explicit LayerTextCallOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }

    const Tensor* content_tensor;
    OP_REQUIRES_OK(context, context->input("content", &content_tensor));

    LayerTextResource* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);

    OP_REQUIRES_OK(context, resource->Write(*content_tensor));
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class LayerTextSyncOp : public OpKernel {
 public:
  explicit LayerTextSyncOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    LayerTextResource* resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);
    OP_REQUIRES_OK(context, resource->Sync());
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>LayerTextInit").Device(DEVICE_CPU),
                        LayerTextInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>LayerTextCall").Device(DEVICE_CPU),
                        LayerTextCallOp);
REGISTER_KERNEL_BUILDER(Name("IO>LayerTextSync").Device(DEVICE_CPU),
                        LayerTextSyncOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
