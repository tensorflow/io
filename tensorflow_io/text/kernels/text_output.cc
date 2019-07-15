/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
namespace {

class TextDatasetOutputOp : public AsyncOpKernel {
 public:
  explicit TextDatasetOutputOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "text_dataset_output") {}

  template <typename T>
  Status ParseScalarArgument(OpKernelContext* ctx,
                             const StringPiece& argument_name, T* output) {
    const Tensor* argument_t;
    TF_RETURN_IF_ERROR(ctx->input(argument_name, &argument_t));
    if (!TensorShapeUtils::IsScalar(argument_t->shape())) {
      return errors::InvalidArgument(argument_name, " must be a scalar");
    }
    *output = argument_t->scalar<T>()();
    return Status::OK();
  }

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
          dataset->MakeIterator(&iter_ctx, "TextDatasetOutputOpIterator", &iterator),
          done);
      std::vector<Tensor> components;
      components.reserve(dataset->output_dtypes().size());
      bool end_of_sequence;
      do {
        OP_REQUIRES_OK_ASYNC(
            ctx, iterator->GetNext(&iter_ctx, &components, &end_of_sequence),
            done);

        if (!end_of_sequence) {
          // TODO (yongtang): In case of TextDataset the shapes of output can
          // only be either a scalar (`batch = 0`) or an 1-D tensor (`batch != 0`)
          // so the following could be simplified by just calling flat<string>.
          // Once other dataset has been introduced for write then additional
          // logic would be neededto handle `batch = 0` vs `batch != 0` case.
          for (int64 i = 0; i < components[0].NumElements(); i++) {
            OP_REQUIRES_OK_ASYNC(
                ctx, file.get()->Append(components[0].flat<string>()(i)), done);
            OP_REQUIRES_OK_ASYNC(
                ctx, file.get()->Append("\n"), done);
          }
        }
        components.clear();
      } while (!end_of_sequence);
          OP_REQUIRES_OK_ASYNC(
              ctx, file.get()->Flush(), done);
      done();
    });
  }

 private:
  BackgroundWorker background_worker_;
};

REGISTER_KERNEL_BUILDER(
    Name("TextDatasetOutput").Device(DEVICE_CPU), TextDatasetOutputOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
