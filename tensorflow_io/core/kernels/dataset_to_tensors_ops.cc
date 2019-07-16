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
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {
namespace {


class DatasetToTensorsOp : public AsyncOpKernel {
 public:
  explicit DatasetToTensorsOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "dataset_to_tensors") {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    // The call to `iterator->GetNext()` may block and depend on an
    // inter-op thread pool thread, so we issue the call from the
    // owned thread pool.
    background_worker_.Schedule([ctx, done]() {
      DatasetBase* dataset;
      OP_REQUIRES_OK_ASYNC(
          ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset), done);
      std::unique_ptr<IteratorBase> iterator;
      IteratorContext::Params params(ctx);
      std::unique_ptr<FunctionHandleCache> function_handle_cache =
          absl::make_unique<FunctionHandleCache>(params.flr);
      params.function_handle_cache = function_handle_cache.get();
      std::unique_ptr<ResourceMgr> resource_mgr =
          absl::make_unique<ResourceMgr>();
      params.resource_mgr = resource_mgr.get();
      IteratorContext iter_ctx(std::move(params));

      OP_REQUIRES_OK_ASYNC(
          ctx,
          dataset->MakeIterator(&iter_ctx, "DatasetToTensorsIterator", &iterator),
          done);

      // NOTE: We must destroy the iterator before calling `done()`, to
      // avoid destruction races.
      IteratorBase* raw_iterator = iterator.release();
      auto cleanup = gtl::MakeCleanup([raw_iterator, done] {
        delete raw_iterator;
        done();
      });

      std::vector<std::vector<Tensor>> components_list;
      bool end_of_sequence = false;


      while (!end_of_sequence) {
        std::vector<Tensor> components;
        components.reserve(dataset->output_dtypes().size());
        components_list.emplace_back(components);
        Status s =
            raw_iterator->GetNext(&iter_ctx, &components_list.back(), &end_of_sequence);
        if (!s.ok()) {
          ctx->SetStatus(s);
          return;
        }
        if (end_of_sequence) {
          components_list.pop_back();
          break;
        }
      }
      if (components_list.empty()) {
        ctx->SetStatus(errors::InvalidArgument("Dataset was empty."));
        return;
      }
      int64 total = 0;
      for (int entry = 0; entry < components_list.size(); entry++) {
        total += components_list[entry][0].shape().dim_size(0);
      }
      for (int i = 0; i < components_list[0].size(); ++i) {
        // TODO: Check that the shapes match the shape attrs.
        TensorShape shape = components_list[0][i].shape();
        shape.set_dim(0, total);
        Tensor* value_tensor;
        Status s = ctx->allocate_output(i, shape, &value_tensor);
        if (!s.ok()) {
          ctx->SetStatus(s);
          return;
        }
        shape.RemoveDim(0);
        Tensor element(components_list[0][i].dtype(), shape);
        int64 index = 0;
        for (int entry = 0; entry < components_list.size(); entry++) {
          for (int64 r = 0; r < components_list[entry][i].shape().dim_size(0); r++) {
            s = batch_util::MaybeMoveSliceToElement(&components_list[entry][i], &element, r);
            s = batch_util::CopyElementToSlice(element, value_tensor, index + r);
          }
          index += components_list[entry][i].shape().dim_size(0);
        }
      }
      components_list.clear();
    });
  }

 private:
  BackgroundWorker background_worker_;
};

REGISTER_KERNEL_BUILDER(Name("DatasetToTensors").Device(DEVICE_CPU),
                        DatasetToTensorsOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
