/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {
namespace data {

template <typename T>
class IOResourceOpKernel : public OpKernel {
 public:
  explicit IOResourceOpKernel(OpKernelConstruction* context) : OpKernel(context) {
    has_resource_type_ = (context->output_type(0) == DT_RESOURCE);
    if (!has_resource_type_) {
      // The resource variant of the op may be placed on non-CPU devices, but
      // this allocation is always on the host. Fortunately we don't need it in
      // the resource case.
      OP_REQUIRES_OK(context,
                     context->allocate_persistent(DT_STRING, TensorShape({2}),
                                                  &handle_, nullptr));
    }
  }

  // The resource is deleted from the resource manager only when it is private
  // to kernel. Ideally the resource should be deleted when it is no longer held
  // by anyone, but it would break backward compatibility.
  ~IOResourceOpKernel() override {
    if (resource_ != nullptr) {
      resource_->Unref();
      if (cinfo_.resource_is_private_to_kernel()) {
        if (!cinfo_.resource_manager()
                 ->template Delete<T>(cinfo_.container(), cinfo_.name())
                 .ok()) {
          // Do nothing; the resource can have been deleted by session resets.
        }
      }
    }
  }

  void Compute(OpKernelContext* context) override TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (resource_ == nullptr) {
      ResourceMgr* mgr = context->resource_manager();
      OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

      T* resource;
      OP_REQUIRES_OK(context,
                     mgr->LookupOrCreate<T>(
                         cinfo_.container(), cinfo_.name(), &resource,
                         [this](T** ret) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                           Status s = CreateResource(ret);
                           if (!s.ok() && *ret != nullptr) {
                             CHECK((*ret)->Unref());
                           }
                           return s;
                         }));

      Status s = VerifyResource(resource);
      if (TF_PREDICT_FALSE(!s.ok())) {
        resource->Unref();
        context->SetStatus(s);
        return;
      }

      if (!has_resource_type_) {
        auto h = handle_.AccessTensor(context)->template flat<tstring>();
        h(0) = cinfo_.container();
        h(1) = cinfo_.name();
      }
      resource_ = resource;
    }
    if (has_resource_type_) {
      OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                                  context, 0, cinfo_.container(), cinfo_.name(),
                                  MakeTypeIndex<T>()));
    } else {
      context->set_output_ref(0, &mu_, handle_.AccessTensor(context));
    }
  }

 protected:
  // Variables accessible from subclasses.
  mutex mu_;
  ContainerInfo cinfo_ TF_GUARDED_BY(mu_);
  T* resource_ TF_GUARDED_BY(mu_) = nullptr;

 private:
  // Must return a T descendant allocated with new that IOResourceOpKernel will
  // take ownership of.
  virtual Status CreateResource(T** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  // During the first Compute(), resource is either created or looked up using
  // shared_name. In the latter case, the resource found should be verified if
  // it is compatible with this op's configuration. The verification may fail in
  // cases such as two graphs asking queues of the same shared name to have
  // inconsistent capacities.
  virtual Status VerifyResource(T* resource) { return Status::OK(); }

  PersistentTensor handle_ TF_GUARDED_BY(mu_);

  // Is the output of the operator of type DT_RESOURCE?
  bool has_resource_type_;
};

}  // namespace data
}  // namespace tensorflow
