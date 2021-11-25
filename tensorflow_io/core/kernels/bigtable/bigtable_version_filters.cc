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
#include "tensorflow_io/core/kernels/bigtable/bigtable_version_filters.h"

namespace cbt = ::google::cloud::bigtable;

namespace tensorflow {
namespace io {

class BigtableLatestFilterOp
    : public AbstractBigtableResourceOp<BigtableFilterResource> {
 public:
  explicit BigtableLatestFilterOp(OpKernelConstruction* ctx)
      : AbstractBigtableResourceOp<BigtableFilterResource>(ctx) {
    VLOG(1) << "BigtableLatestFilterOp ctor ";
  }

 private:
  StatusOr<BigtableFilterResource*> CreateResource() override {
    return new BigtableFilterResource(cbt::Filter::Latest(1));
  }
};

REGISTER_KERNEL_BUILDER(Name("BigtableLatestFilter").Device(DEVICE_CPU),
                        BigtableLatestFilterOp);

class BigtableTimestampRangeFilterOp
    : public AbstractBigtableResourceOp<BigtableFilterResource> {
 public:
  explicit BigtableTimestampRangeFilterOp(OpKernelConstruction* ctx)
      : AbstractBigtableResourceOp<BigtableFilterResource>(ctx) {
    VLOG(1) << "BigtableTimestampRangeFilterOp ctor ";
    OP_REQUIRES_OK(ctx, ctx->GetAttr("start_ts_us", &start_ts_us_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("end_ts_us", &end_ts_us_));
  }

 private:
  StatusOr<BigtableFilterResource*> CreateResource() override {
    return new BigtableFilterResource(
        cbt::Filter::TimestampRangeMicros(start_ts_us_, end_ts_us_));
  }

 private:
  int64_t start_ts_us_;
  int64_t end_ts_us_;
};

REGISTER_KERNEL_BUILDER(Name("BigtableTimestampRangeFilter").Device(DEVICE_CPU),
                        BigtableTimestampRangeFilterOp);

class BigtablePrintFilterOp : public OpKernel {
 public:
  explicit BigtablePrintFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    BigtableFilterResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "filter", &resource));
    core::ScopedUnref unref(resource);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output_tensor));
    auto output_v = output_tensor->tensor<tstring, 1>();

    output_v(0) = resource->ToString();
  }
};

REGISTER_KERNEL_BUILDER(Name("BigtablePrintFilter").Device(DEVICE_CPU),
                        BigtablePrintFilterOp);

}  // namespace io
}  // namespace tensorflow
