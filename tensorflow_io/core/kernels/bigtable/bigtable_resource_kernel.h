/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef BIGTABLE_RESOURCE_KERNEL_H
#define BIGTABLE_RESOURCE_KERNEL_H

#include "absl/memory/memory.h"
#include "google/cloud/bigtable/table.h"
#include "google/cloud/bigtable/table_admin.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {
namespace io {

template <typename T>
class AbstractBigtableResourceOp : public OpKernel {
 public:
  explicit AbstractBigtableResourceOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    ResourceMgr* mgr = context->resource_manager();
    ContainerInfo cinfo;
    OP_REQUIRES_OK(context, cinfo.Init(mgr, def()));

    StatusOr<T*> maybe_resource = CreateResource();
    OP_REQUIRES_OK(context, maybe_resource.status());

    OP_REQUIRES_OK(context, mgr->Create<T>(cinfo.container(), cinfo.name(),
                                           maybe_resource.ValueOrDie()));

    OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                                context, 0, cinfo.container(), cinfo.name(),
                                TypeIndex::Make<T>()));
  }

 private:
  virtual StatusOr<T*> CreateResource() = 0;
};

}  // namespace io
}  // namespace tensorflow

#endif /* BIGTABLE_RESOURCE_KERNEL_H */
