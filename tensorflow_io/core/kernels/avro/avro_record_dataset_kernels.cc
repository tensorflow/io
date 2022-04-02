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
#include "tensorflow_io/core/kernels/avro/avro_record_dataset_kernels.h"

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow_io/core/kernels/avro/utils/avro_record_reader.h"
// TODO(fraudies): Wait until TF tensorflow/core/kernels/data/name_utils.h is
// visible
#include "tensorflow_io/core/kernels/avro/utils/name_utils.h"

namespace tensorflow {
namespace data {

// https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/core/kernels/data/tf_record_dataset_op.cc

/* static */ constexpr const char* const AvroRecordDatasetOp::kDatasetType;
/* static */ constexpr const char* const AvroRecordDatasetOp::kFileNames;
/* static */ constexpr const char* const AvroRecordDatasetOp::kBufferSize;
/* static */ constexpr const char* const AvroRecordDatasetOp::kReaderSchema;

constexpr char kCurrentFileIndex[] = "current_file_index";
constexpr char kOffset[] = "offset";

class AvroRecordDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx, std::vector<tstring> filenames,
                   int64 buffer_size, const tstring& reader_schema)
      : DatasetBase(DatasetContext(ctx)),
        filenames_(std::move(filenames)),
        options_(AvroReaderOptions::CreateReaderOptions()) {
    if (buffer_size > 0) {
      options_.buffer_size = buffer_size;
    }
    if (!reader_schema.empty()) {
      options_.reader_schema = reader_schema;
    }
    VLOG(7) << "Created dataset with first filename: " << filenames_[0];
    VLOG(7) << "Buffer size " << buffer_size / 1024 << "kBy";
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const {
    return Status::OK();
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
    static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
    return *dtypes;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    static std::vector<PartialTensorShape>* shapes =
        new std::vector<PartialTensorShape>({{}});
    return *shapes;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status CheckExternalState() const override { return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* filenames = nullptr;
    TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
    Node* buffer_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(options_.buffer_size, &buffer_size));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      do {
        // We are currently processing a file, so try to read the next record.
        if (reader_) {
          out_tensors->emplace_back(ctx->allocator({}), DT_STRING,
                                    TensorShape({}));
          Status s =
              reader_->ReadRecord(&out_tensors->back().scalar<tstring>()());
          if (s.ok()) {
            *end_of_sequence = false;
            return Status::OK();
          }
          out_tensors->pop_back();
          if (!errors::IsOutOfRange(s)) {
            // In case of other errors e.g., DataLoss, we still move forward
            // the file index so that it works with ignore_errors.
            // Otherwise the same file will repeat.
            ResetStreamsLocked();
            ++current_file_index_;
            return s;
          }

          // We have reached the end of the current file, so maybe move on to
          // next file.
          ResetStreamsLocked();
          ++current_file_index_;
        }

        // Iteration ends when there are no more files to process.
        if (current_file_index_ == dataset()->filenames_.size()) {
          *end_of_sequence = true;
          return Status::OK();
        }

        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      } while (true);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      return errors::Unimplemented("SaveInternal");
    }

    /*
    Status SaveInternal(IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurrentFileIndex),
                                             current_file_index_));

      if (reader_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kOffset), reader_->TellOffset()));
      }
      return Status::OK();
    }
    */

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return errors::Unimplemented(
          "Iterator does not support 'RestoreInternal')");
    }

    /*
    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      ResetStreamsLocked();
      int64 current_file_index;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurrentFileIndex),
                                            &current_file_index));
      current_file_index_ = size_t(current_file_index);
      if (reader->Contains(full_name(kOffset))) {
        int64 offset;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kOffset), &offset));
        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        TF_RETURN_IF_ERROR(reader_->SeekOffset(offset));
      }
      return Status::OK();
    }
    */

   private:
    // Sets up reader streams to read from the file at `current_file_index_`.
    Status SetupStreamsLocked(Env* env) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (current_file_index_ >= dataset()->filenames_.size()) {
        return errors::InvalidArgument(
            "current_file_index_:", current_file_index_,
            " >= filenames_.size():", dataset()->filenames_.size());
      }

      // Actually move on to next file.
      const string& next_filename = dataset()->filenames_[current_file_index_];
      TF_RETURN_IF_ERROR(env->NewRandomAccessFile(next_filename, &file_));
      reader_ = absl::make_unique<SequentialAvroRecordReader>(
          file_.get(), dataset()->options_);
      return Status::OK();
    }

    // Resets all reader streams.
    void ResetStreamsLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      reader_.reset();
      file_.reset();
    }

    mutex mu_;
    size_t current_file_index_ TF_GUARDED_BY(mu_) = 0;

    // `reader_` will borrow the object that `file_` points to, so
    // we must destroy `reader_` before `file_`.
    std::unique_ptr<RandomAccessFile> file_ TF_GUARDED_BY(mu_);
    std::unique_ptr<SequentialAvroRecordReader> reader_ TF_GUARDED_BY(mu_);
  };

  const std::vector<tstring> filenames_;
  AvroReaderOptions options_;
};

AvroRecordDatasetOp::AvroRecordDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {}

void AvroRecordDatasetOp::MakeDataset(OpKernelContext* ctx,
                                      DatasetBase** output) {
  const Tensor* filenames_tensor;
  OP_REQUIRES_OK(ctx, ctx->input(kFileNames, &filenames_tensor));
  OP_REQUIRES(
      ctx, filenames_tensor->dims() <= 1,
      errors::InvalidArgument("`filenames` must be a scalar or a vector."));

  std::vector<tstring> filenames;
  filenames.reserve(filenames_tensor->NumElements());
  for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
    VLOG(2) << "Reading file: " << filenames_tensor->flat<tstring>()(i);
    filenames.push_back(filenames_tensor->flat<tstring>()(i));
  }

  int64 buffer_size = -1;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(ctx, buffer_size >= 0,
              errors::InvalidArgument(
                  "`buffer_size` must be >= 0 (0 == no buffering)"));
  tstring reader_schema = "";
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument<tstring>(ctx, kReaderSchema, &reader_schema));

  *output = new Dataset(ctx, std::move(filenames), buffer_size, reader_schema);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("IO>AvroRecordDataset").Device(DEVICE_CPU),
                        AvroRecordDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
