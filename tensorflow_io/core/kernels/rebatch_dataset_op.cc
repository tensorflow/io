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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {
namespace {

class AdjustBatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit AdjustBatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 batch_size = 0;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "batch_size", &batch_size));
    OP_REQUIRES(
        ctx, batch_size > 0,
        errors::InvalidArgument("Batch size must be greater than zero."));

    string batch_mode = "";
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<string>(ctx, "batch_mode", &batch_mode));
    OP_REQUIRES(
        ctx, (batch_mode == "" ||
             batch_mode == "keep" ||
             batch_mode == "drop" ||
             batch_mode == "pad"), errors::InvalidArgument("invalid batch_mode: ", batch_mode));


    *output =
        new Dataset(ctx, batch_size, batch_mode, input);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, int64 batch_size, string batch_mode,
            const DatasetBase* input)
        : DatasetBase(DatasetContext(ctx)),
          batch_size_(batch_size),
          batch_mode_(batch_mode),
          input_(input) {
      input_->Ref();

      const auto& input_shapes = input_->output_shapes();
      output_shapes_.reserve(input_shapes.size());
      // Always set the first dim as None unless batch_mode is specified.
      for (const auto& input_shape : input_shapes) {
        if (!input_shape.unknown_rank()) {
          output_shapes_.emplace_back(
              PartialTensorShape({-1}).Concatenate(input_shape));
          output_shapes_.back().RemoveDim(1);
        } else {
          output_shapes_.emplace_back();
        }
      }
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Rebatch")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return strings::StrCat("AdjustBatchDatasetOp(", batch_size_, ")::Dataset");
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* batch_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));
      Node* batch_mode = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_mode_, &batch_mode));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {input_graph_node, batch_size, batch_mode},
                        output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            current_index_(0),
            current_batch_size_(0) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (!input_impl_) {
          *end_of_sequence = true;
          return Status::OK();
        }
        *end_of_sequence = false;

        int64 chunk_read = 0;

        out_tensors->clear();
        std::vector<Tensor> elements;
        while (!*end_of_sequence) {
          if (current_index_ < current_batch_size_) {
            if (out_tensors->size() == 0) {
              out_tensors->reserve(tensors_.size());
              elements.reserve(tensors_.size());
              for (size_t i = 0; i < tensors_.size(); ++i) {
                TensorShape shape = tensors_[i].shape();
                shape.RemoveDim(0);
                elements.emplace_back(ctx->allocator({}), tensors_[i].dtype(), shape);
                shape.InsertDim(0, dataset()->batch_size_);
                out_tensors->emplace_back(ctx->allocator({}), tensors_[i].dtype(), shape);
              }
            }
            if (out_tensors->size() != tensors_.size()) {
              return errors::InvalidArgument("number tensors should match previous one, ", tensors_.size(), " vs. ", out_tensors->size());
            }
            int64 chunk_to_read = (current_batch_size_ - current_index_) < (dataset()->batch_size_ - chunk_read) ? (current_batch_size_ - current_index_) : (dataset()->batch_size_ - chunk_read);
            for (int i = 0; i < tensors_.size(); ++i) {
              // TODO: concurrent copy?
              for (int64 r = 0; r < chunk_to_read; ++r) {
                TF_RETURN_IF_ERROR(batch_util::MaybeMoveSliceToElement(
                    &tensors_[i], &elements[i], current_index_ + r));
                TF_RETURN_IF_ERROR(batch_util::CopyElementToSlice(
                    elements[i], &(*out_tensors)[i], chunk_read + r));
              }
            }
            chunk_read += chunk_to_read;
            current_index_ += chunk_to_read;
            if (chunk_read == dataset()->batch_size_) {
              *end_of_sequence = false;
              return Status::OK();
            }
          }
          current_index_ = 0;
          current_batch_size_ = 0;
          tensors_.clear();
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, &tensors_, end_of_sequence));
          if (!*end_of_sequence) {
            for (size_t i = 0; i < tensors_.size(); ++i) {
              if (tensors_[i].dims() == 0) {
                return errors::InvalidArgument(
                    "Input element must have a non-scalar value in each "
                    "component.");
              }
              if (tensors_[i].dim_size(0) != tensors_[0].dim_size(0)) {
                return errors::InvalidArgument(
                    "Input element must have the same batch size in each "
                    "component. Component 0 had size ",
                    tensors_[0].dim_size(0), " but component ", i,
                    " had size, ", tensors_[i].dim_size(0), ".");
              }
            }
            current_batch_size_ = tensors_[0].dim_size(0);
          }
        }
        // Finally, resize if needed
        if (chunk_read > 0) {
          if (chunk_read < dataset()->batch_size_) {
            // "keep" reminder will need to resize
            if (dataset()->batch_mode_ == "" || dataset()->batch_mode_ == "keep") {
              for (int i = 0; i < out_tensors->size(); ++i) {
                TensorShape shape = (*out_tensors)[i].shape();
                shape.set_dim(0, chunk_read);
                Tensor value_tensor(ctx->allocator({}), (*out_tensors)[i].dtype(), shape);
                for (int64 r = 0; r < chunk_read; r++) {
                  TF_RETURN_IF_ERROR(batch_util::MaybeMoveSliceToElement(
                      &(*out_tensors)[i], &elements[i], r));
                  TF_RETURN_IF_ERROR(batch_util::CopyElementToSlice(
                      elements[i], &value_tensor, r));
                }
                (*out_tensors)[i] = std::move(value_tensor);
              }
            // "drop" the reminder
            } else if (dataset()->batch_mode_ == "drop") {
              out_tensors->clear();
              input_impl_.reset();
              *end_of_sequence = true;
              return Status::OK();
            }
            // otherwise "pad" means keep the size
            // TODO:  at the moment the remining of the Tensor will
            // be filled with default values, so there is nothing
            // needs to be done. If non-default values are needed
            // then it will need to be filled.
          }
          *end_of_sequence = false;
          return Status::OK();
        }
        out_tensors->clear();
        input_impl_.reset();
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         dataset()->batch_size_);
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        return errors::Unimplemented("SaveInternal is currently not supported");
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return errors::Unimplemented("RestoreInternal is currently not supported");
      }

     private:
      mutex mu_;
      int64 current_index_ GUARDED_BY(mu_);
      int64 current_batch_size_ GUARDED_BY(mu_);
      std::vector<Tensor> tensors_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const int64 batch_size_;
    const string batch_mode_;
    const DatasetBase* const input_;
    std::vector<PartialTensorShape> output_shapes_;
  };
};

REGISTER_KERNEL_BUILDER(Name("AdjustBatchDataset").Device(DEVICE_CPU),
                        AdjustBatchDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
