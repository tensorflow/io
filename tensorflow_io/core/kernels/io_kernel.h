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
#include "tensorflow/core/lib/strings/scanner.h"

namespace tensorflow {
namespace data {

template <typename T>
class IOResourceOpKernel : public OpKernel {
 public:
  explicit IOResourceOpKernel(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
    OP_REQUIRES_OK(context, context->GetAttr("container", &container_));
    OP_REQUIRES(context,
                (container_.empty() || IsValidContainerName(container_)),
                errors::InvalidArgument(
                    "container contains invalid characters: ", container_));
  }

  virtual ~IOResourceOpKernel() {}

  virtual Status ResourceKernel(OpKernelContext* context, T* resource) {
    return Status::OK();
  }

  void Compute(OpKernelContext* context) override TF_LOCKS_EXCLUDED(mu_) {
    const Tensor* shared_tensor;

    OP_REQUIRES_OK(context, context->input("shared", &shared_tensor));
    string shared = shared_tensor->scalar<tstring>()();
    if (shared.empty()) {
      // TODO: no resource manager case
      OP_REQUIRES_OK(
          context, errors::InvalidArgument("shared cannot be empty: ", shared));
    }

    OP_REQUIRES(
        context, (shared[0] != '_'),
        errors::InvalidArgument("shared cannot start with '_':", shared));

    mutex_lock l(mu_);
    ResourceMgr* mgr = context->resource_manager();

    T* resource;
    OP_REQUIRES_OK(
        context,
        mgr->LookupOrCreate<T>(
            (container_.empty() ? container_ : mgr->default_container()),
            shared, &resource,
            [this, context](T** ret) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
              const Tensor* input_tensor;
              TF_RETURN_IF_ERROR(context->input("input", &input_tensor));
              string input = input_tensor->scalar<tstring>()();
              *ret = new T(env_);
              if ((*ret) != nullptr) {
                Status s = (*ret)->Init(input);
                if (!s.ok()) {
                  CHECK((*ret)->Unref());
                }
                return s;
              }
              return errors::InvalidArgument(
                  "unable to allocate memory for resource");
            }));
    OP_REQUIRES_OK(context, ResourceKernel(context, resource));
  }

 protected:
  mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  string container_ TF_GUARDED_BY(mu_);

 private:
  static bool IsValidContainerName(StringPiece s) {
    using ::tensorflow::strings::Scanner;
    return Scanner(s)
        .One(Scanner::LETTER_DIGIT_DOT)
        .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH)
        .Eos()
        .GetResult();
  }
};

}  // namespace data
}  // namespace tensorflow
