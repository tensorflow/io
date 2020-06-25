/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {
namespace data {

template <typename T>
class DatasetOutputOp : public AsyncOpKernel {
 public:
  explicit DatasetOutputOp<T>(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "text_dataset_output") {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    // The call to `iterator->GetNext()` may block and depend on an inter-op
    // thread pool thread, so we issue the call using a background thread.
    background_worker_.Schedule([this, ctx, done]() {
      string filename;
      OP_REQUIRES_OK_ASYNC(
          ctx, ParseScalarArgument<string>(ctx, "filename", &filename), done);
      std::unique_ptr<WritableFile> file;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->env()->NewWritableFile(filename, &file),
                           done);

      DatasetBase* dataset;
      OP_REQUIRES_OK_ASYNC(
          ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset), done);
      std::unique_ptr<IteratorBase> iterator;
      IteratorContext::Params params(ctx);
      std::unique_ptr<FunctionHandleCache> function_handle_cache =
          absl::make_unique<FunctionHandleCache>(params.flr);
      params.function_handle_cache = function_handle_cache.get();
      IteratorContext iter_ctx(std::move(params));

      OP_REQUIRES_OK_ASYNC(
          ctx,
          dataset->MakeIterator(&iter_ctx, "TextDatasetOutputOpIterator",
                                &iterator),
          done);
      std::vector<Tensor> components;
      components.reserve(dataset->output_dtypes().size());
      bool end_of_sequence;
      std::unique_ptr<T> output;
      do {
        OP_REQUIRES_OK_ASYNC(
            ctx, iterator->GetNext(&iter_ctx, &components, &end_of_sequence),
            done);

        if (!end_of_sequence) {
          OP_REQUIRES_OK_ASYNC(ctx, output.get()->Write(file.get(), components),
                               done);
        }
        components.clear();
      } while (!end_of_sequence);
      OP_REQUIRES_OK_ASYNC(ctx, output.get()->Final(file.get()), done);
      done();
    });
  }

 private:
  BackgroundWorker background_worker_;
};

}  // namespace data
}  // namespace tensorflow
