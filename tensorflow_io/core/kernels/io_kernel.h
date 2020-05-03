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

  virtual ~IOResourceOpKernel() {
    // TODO: LRU cache with adjustable size?
    if (std::get<0>(resource_created_) != "") {
      const string& shared = std::get<0>(resource_created_);
      T* resource = std::get<1>(resource_created_);
      ResourceMgr* mgr = std::get<2>(resource_created_);
      Status status = mgr->template Delete<T>(
          (container_.empty() ? container_ : mgr->default_container()), shared);
      if (!status.ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

  // ResourceKernel needs to be implemented by subclass
  virtual Status ResourceKernel(OpKernelContext* context, T* resource) = 0;

  void Compute(OpKernelContext* context) override TF_LOCKS_EXCLUDED(mu_) {
    const Tensor* shared_tensor;

    OP_REQUIRES_OK(context, context->input("shared", &shared_tensor));
    string shared = shared_tensor->scalar<tstring>()();
    if (shared.empty()) {
      // TODO: non-resource manager  case
      OP_REQUIRES_OK(
          context, errors::InvalidArgument("shared cannot be empty: ", shared));
    }

    OP_REQUIRES(
        context, (shared[0] != '_'),
        errors::InvalidArgument("shared cannot start with '_':", shared));

    mutex_lock g(mu_);
    T* resource;
    // TODO: LRU cache with adjustable size?
    if (std::get<0>(resource_created_) != "" &&
        std::get<0>(resource_created_) != shared) {
      const string& shared = std::get<0>(resource_created_);
      T* resource = std::get<1>(resource_created_);
      ResourceMgr* mgr = std::get<2>(resource_created_);
      Status status = mgr->template Delete<T>(
          (container_.empty() ? container_ : mgr->default_container()), shared);
      if (!status.ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
      resource_created_ =
          std::tuple<string, T*, ResourceMgr*>{"", nullptr, nullptr};
    }

    ResourceMgr* mgr = context->resource_manager();
    OP_REQUIRES_OK(
        context,
        mgr->LookupOrCreate<T>(
            (container_.empty() ? container_ : mgr->default_container()),
            shared, &resource,
            [this, context, mgr](T** ret) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
              const Tensor* input_tensor;
              TF_RETURN_IF_ERROR(context->input("input", &input_tensor));
              string input = input_tensor->scalar<tstring>()();
              *ret = new T(env_);
              if ((*ret) != nullptr) {
                Status status = (*ret)->Init(input);
                if (!status.ok()) {
                  CHECK((*ret)->Unref());
                }
                if (status.ok()) {
                  resource_created_ =
                      std::tuple<string, T*, ResourceMgr*>{input, (*ret), mgr};
                }
                return status;
              }
              return errors::InvalidArgument(
                  "unable to allocate memory for resource");
            }));
    core::ScopedUnref unref(resource);
    OP_REQUIRES_OK(context, ResourceKernel(context, resource));
  }

 protected:
  static mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  string container_ TF_GUARDED_BY(mu_);
  // Resource created by kernel, and the kernel is responsible for deletion.
  // In case new resource is created and overrides old one, old one must be
  // deleted as well.
  std::tuple<string, T*, ResourceMgr*> resource_created_ TF_GUARDED_BY(mu_);

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

template <typename T>
mutex IOResourceOpKernel<T>::mu_(LINKER_INITIALIZED);
}  // namespace data
}  // namespace tensorflow
