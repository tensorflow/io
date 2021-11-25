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
#include "tensorflow_io/core/kernels/bigtable/bigtable_row_range.h"

namespace cbt = ::google::cloud::bigtable;

namespace tensorflow {
namespace io {

class BigtableEmptyRowRangeOp
    : public AbstractBigtableResourceOp<BigtableRowRangeResource> {
 public:
  explicit BigtableEmptyRowRangeOp(OpKernelConstruction* ctx)
      : AbstractBigtableResourceOp<BigtableRowRangeResource>(ctx) {
    VLOG(1) << "BigtableEmptyRowRangeOp ctor ";
  }

 private:
  StatusOr<BigtableRowRangeResource*> CreateResource() override {
    return new BigtableRowRangeResource(cbt::RowRange::Empty());
  }
};

REGISTER_KERNEL_BUILDER(Name("BigtableEmptyRowRange").Device(DEVICE_CPU),
                        BigtableEmptyRowRangeOp);

class BigtableRowRangeOp
    : public AbstractBigtableResourceOp<BigtableRowRangeResource> {
 public:
  explicit BigtableRowRangeOp(OpKernelConstruction* ctx)
      : AbstractBigtableResourceOp<BigtableRowRangeResource>(ctx) {
    VLOG(1) << "BigtableRowRangeOp ctor ";
    OP_REQUIRES_OK(ctx, ctx->GetAttr("left_row_key", &left_row_key_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("left_open", &left_open_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("right_row_key", &right_row_key_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("right_open", &right_open_));
  }

 private:
  StatusOr<BigtableRowRangeResource*> CreateResource() override {
    VLOG(1) << "BigtableRowRangeOp constructing row_range:"
            << (left_open_ ? "(" : "[") << left_row_key_ << ":"
            << right_row_key_ << (right_open_ ? ")" : "]");

    // both empty - infinite
    if (left_row_key_.empty() && right_row_key_.empty()) {
      return new BigtableRowRangeResource(cbt::RowRange::InfiniteRange());
    }

    // open
    if (left_open_ && right_open_) {
      return new BigtableRowRangeResource(
          cbt::RowRange::Open(left_row_key_, right_row_key_));
    }
    // closed
    if (!left_open_ && !right_open_) {
      return new BigtableRowRangeResource(
          cbt::RowRange::Closed(left_row_key_, right_row_key_));
    }
    // right_open
    if (!left_open_ && right_open_) {
      return new BigtableRowRangeResource(
          cbt::RowRange::RightOpen(left_row_key_, right_row_key_));
    }
    // left_open
    if (left_open_ && !right_open_) {
      return new BigtableRowRangeResource(
          cbt::RowRange::LeftOpen(left_row_key_, right_row_key_));
    }
    return Status(
        error::INTERNAL,
        "Reached impossible branch. Above clauses should cover all possible "
        "values of left_open_ and right_open_. Please report this issue here:"
        "https://github.com/tensorflow/io/issues/new/choose quoting these"
        "values: left_open:" +
            std::to_string(left_open_) +
            " right_open:" + std::to_string(right_open_));
  }

 private:
  mutex mu_;
  std::string left_row_key_ TF_GUARDED_BY(mu_);
  bool left_open_ TF_GUARDED_BY(mu_);
  std::string right_row_key_ TF_GUARDED_BY(mu_);
  bool right_open_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("BigtableRowRange").Device(DEVICE_CPU),
                        BigtableRowRangeOp);

class BigtablePrintRowRangeOp : public OpKernel {
 public:
  explicit BigtablePrintRowRangeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    BigtableRowRangeResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "row_range", &resource));
    core::ScopedUnref unref(resource);

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output_tensor));
    auto output_v = output_tensor->tensor<tstring, 1>();

    output_v(0) = resource->ToString();
  }
};

REGISTER_KERNEL_BUILDER(Name("BigtablePrintRowRange").Device(DEVICE_CPU),
                        BigtablePrintRowRangeOp);

class BigtablePrefixRowRangeOp
    : public AbstractBigtableResourceOp<BigtableRowRangeResource> {
 public:
  explicit BigtablePrefixRowRangeOp(OpKernelConstruction* ctx)
      : AbstractBigtableResourceOp<BigtableRowRangeResource>(ctx) {
    VLOG(1) << "BigtablePrefixRowRangeOp ctor ";
    OP_REQUIRES_OK(ctx, ctx->GetAttr("prefix", &prefix_));
  }

 private:
  StatusOr<BigtableRowRangeResource*> CreateResource() override {
    return new BigtableRowRangeResource(cbt::RowRange::Prefix(prefix_));
  }

 private:
  std::string prefix_;
};

REGISTER_KERNEL_BUILDER(Name("BigtablePrefixRowRange").Device(DEVICE_CPU),
                        BigtablePrefixRowRangeOp);

}  // namespace io
}  // namespace tensorflow
