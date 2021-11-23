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
#include "tensorflow_io/core/kernels/bigtable/bigtable_row_set.h"

namespace cbt = ::google::cloud::bigtable;

namespace tensorflow {
namespace io {

class BigtableEmptyRowSetOp
    : public AbstractBigtableResourceOp<BigtableRowSetResource> {
 public:
  explicit BigtableEmptyRowSetOp(OpKernelConstruction* ctx)
      : AbstractBigtableResourceOp<BigtableRowSetResource>(ctx) {
    VLOG(1) << "BigtableEmptyRowSetOp ctor ";
  }

 private:
  StatusOr<BigtableRowSetResource*> CreateResource() override {
    return new BigtableRowSetResource(cbt::RowSet());
  }
};

REGISTER_KERNEL_BUILDER(Name("BigtableEmptyRowSet").Device(DEVICE_CPU),
                        BigtableEmptyRowSetOp);

class BigtablePrintRowSetOp : public OpKernel {
 public:
  explicit BigtablePrintRowSetOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    BigtableRowSetResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "row_set", &resource));
    core::ScopedUnref unref(resource);

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output_tensor));
    auto output_v = output_tensor->tensor<tstring, 1>();

    output_v(0) = resource->ToString();
  }
};

REGISTER_KERNEL_BUILDER(Name("BigtablePrintRowSet").Device(DEVICE_CPU),
                        BigtablePrintRowSetOp);

class BigtableRowSetAppendRowOp : public OpKernel {
 public:
  explicit BigtableRowSetAppendRowOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("row_key", &row_key_));
  }

  void Compute(OpKernelContext* context) override {
    BigtableRowSetResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "row_set", &resource));
    core::ScopedUnref unref(resource);

    resource->AppendRow(row_key_);
  }

 private:
  std::string row_key_;
};

REGISTER_KERNEL_BUILDER(Name("BigtableRowSetAppendRow").Device(DEVICE_CPU),
                        BigtableRowSetAppendRowOp);

class BigtableRowSetAppendRowRangeOp : public OpKernel {
 public:
  explicit BigtableRowSetAppendRowRangeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    mutex_lock lock(mu_);
    BigtableRowSetResource* row_set_resource;
    OP_REQUIRES_OK(
        context, GetResourceFromContext(context, "row_set", &row_set_resource));
    core::ScopedUnref row_set_resource_unref(row_set_resource);

    BigtableRowRangeResource* row_range_resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "row_range",
                                                   &row_range_resource));
    core::ScopedUnref row_range_resource_unref(row_range_resource);

    row_set_resource->AppendRowRange(row_range_resource->row_range());
  }

 private:
  mutex mu_;
  std::string row_key_;
};

REGISTER_KERNEL_BUILDER(Name("BigtableRowSetAppendRowRange").Device(DEVICE_CPU),
                        BigtableRowSetAppendRowRangeOp);

class BigtableRowSetIntersectOp : public OpKernel {
 public:
  explicit BigtableRowSetIntersectOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    ResourceMgr* mgr = context->resource_manager();
    OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

    BigtableRowSetResource* row_set_resource;
    OP_REQUIRES_OK(
        context, GetResourceFromContext(context, "row_set", &row_set_resource));
    core::ScopedUnref row_set_resource_unref(row_set_resource);

    BigtableRowRangeResource* row_range_resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "row_range",
                                                   &row_range_resource));
    core::ScopedUnref row_range_resource_unref(row_range_resource);

    BigtableRowSetResource* result_resource = new BigtableRowSetResource(
        row_set_resource->Intersect(row_range_resource->row_range()));

    OP_REQUIRES_OK(context,
                   mgr->Create<BigtableRowSetResource>(
                       cinfo_.container(), cinfo_.name(), result_resource));

    OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                                context, 0, cinfo_.container(), cinfo_.name(),
                                TypeIndex::Make<BigtableRowSetResource>()));
  }

 protected:
  mutex mu_;
  ContainerInfo cinfo_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("BigtableRowSetIntersect").Device(DEVICE_CPU),
                        BigtableRowSetIntersectOp);

}  // namespace io
}  // namespace tensorflow
