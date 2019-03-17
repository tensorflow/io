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

namespace tensorflow {
class OutputSequence : public ResourceBase {
 public:
  OutputSequence(Env* env)
   : env_(env) {}

  virtual ~OutputSequence() override {}
  virtual Status Initialize(const std::vector<string>& destination) {
    destination_ = destination;
    return Status::OK();
  }
  virtual Status SetItem(int64 index, const void *item) = 0;
  virtual string DebugString() const {
    return strings::StrCat("OutputSequence[]");
  }
 protected:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::vector<string> destination_ GUARDED_BY(mu_);
};

}  // namespace tensorflow
