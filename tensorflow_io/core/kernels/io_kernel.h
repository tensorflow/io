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

#include "tensorflow/core/framework/op_kernel.h"
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

  // ResourceKernel needs to be implemented by subclass
  virtual Status ResourceKernel(OpKernelContext* context, T* resource) = 0;

  virtual void Compute(OpKernelContext* context) override {
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

    std::shared_ptr<T> resource;
    {
      mutex_lock l(mu_);
      // TODO: LRU cache with adjustable size?
      if (resource_created_ != "" && resource_created_ != shared) {
        entries_.erase(container_ + "/" + resource_created_);
        resource_created_ = "";
      }

      auto lookup = entries_.find(container_ + "/" + shared);
      if (lookup == entries_.end()) {
        const Tensor* input_tensor;
        OP_REQUIRES_OK(context, context->input("input", &input_tensor));
        string input = input_tensor->scalar<tstring>()();

        resource.reset(new T(env_));
        OP_REQUIRES_OK(context, resource->Init(input));
        entries_[container_ + "/" + shared] = resource;
        resource_created_ = shared;
      } else {
        resource = lookup->second;
      }
    }
    OP_REQUIRES_OK(context, ResourceKernel(context, resource.get()));
  }

 protected:
  static mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  string container_ TF_GUARDED_BY(mu_);
  // Resource created by kernel, and the kernel is responsible for deletion.
  // In case new resource is created and overrides old one, old one must be
  // deleted as well.
  string resource_created_ TF_GUARDED_BY(mu_);

  static std::unordered_map<string, std::shared_ptr<T>> entries_
      TF_GUARDED_BY(mu_);

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

template <typename T>
std::unordered_map<string, std::shared_ptr<T>> IOResourceOpKernel<T>::entries_(
    LINKER_INITIALIZED);
}  // namespace data
}  // namespace tensorflow
