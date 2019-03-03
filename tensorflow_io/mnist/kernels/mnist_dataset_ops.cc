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
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"

namespace tensorflow {
namespace data {
namespace {

static const int64 kMNISTBufferSize = 128 * 1024;

class MNISTDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;
  explicit MNISTDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
    op_ = ctx->def().op();
  }
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));

    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }
    string compression_type;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "compression_type",
                                                    &compression_type));
    OP_REQUIRES(ctx,
                compression_type.empty() || compression_type == "ZLIB" || compression_type == "GZIP",
                errors::InvalidArgument("Unsupported compression_type."));
    *output = new Dataset(ctx, filenames, compression_type, op_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const std::vector<string>& filenames, const string& compression_type, const string& op)
        : DatasetBase(DatasetContext(ctx)),
          filenames_(filenames),
          compression_type_(compression_type),
	  op_(op) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      if (op_ == "MNISTImageDataset") {
        return std::unique_ptr<IteratorBase>(new ImageIterator({this, strings::StrCat(prefix, "::MNIST")}));
      }
      return std::unique_ptr<IteratorBase>(new LabelIterator({this, strings::StrCat(prefix, "::MNIST")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_UINT8});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{-1, -1}});
      return *shapes;
    }

    string DebugString() const override {
      return "MNISTDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      Node* compression_type = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(compression_type_, &compression_type));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {filenames, compression_type}, output));
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
          if (buffered_input_stream_) {
            if (index_ < count_) {
	      TF_RETURN_IF_ERROR(ReadRecord(ctx, out_tensors));

	      index_++;

              *end_of_sequence = false;
              return Status::OK();
	    }
            // We have reached the end of the current file, so maybe
            // move on to next file.
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
      Status SaveInternal(IteratorStateWriter* writer) override {
        return errors::Unimplemented("SaveInternal is currently not supported");
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return errors::Unimplemented(
            "RestoreInternal is currently not supported");
      }
      virtual Status ReadHeader() = 0;
      virtual Status ReadRecord(IteratorContext* ctx, std::vector<Tensor>* out_tensors) = 0;
     private:
      // Sets up streams to read from the topic at
      // `current_file_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_file_index_ >= dataset()->filenames_.size()) {
          return errors::InvalidArgument(
              "current_file_index_:", current_file_index_,
              " >= filenames_.size():", dataset()->filenames_.size());
        }

        // Actually move on to next file.
        const string& filename = dataset()->filenames_[current_file_index_];
        TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file_));
        if (!dataset()->compression_type_.empty()) {
          const io::ZlibCompressionOptions zlib_options =
              dataset()->compression_type_ == "ZLIB"
                  ? io::ZlibCompressionOptions::DEFAULT()
                  : io::ZlibCompressionOptions::GZIP();
          file_stream_.reset(new io::RandomAccessInputStream(file_.get()));
          buffered_input_stream_.reset(new io::ZlibInputStream(
              file_stream_.get(), kMNISTBufferSize,
              kMNISTBufferSize, zlib_options));
        } else {
          buffered_input_stream_.reset(new io::BufferedInputStream(
              file_.get(), kMNISTBufferSize));
        }

	return ReadHeader();
      }

      // Resets file streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        buffered_input_stream_.reset();
        file_.reset();
        index_ = 0;
        count_ = 0;
        rows_ = 0;
        cols_ = 0;
      }

     protected:
      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<RandomAccessFile> file_ GUARDED_BY(mu_);  // must outlive buffered_input_stream_
      std::unique_ptr<io::RandomAccessInputStream> file_stream_;  // must outlive buffered_input_stream_
      std::unique_ptr<io::InputStreamInterface> buffered_input_stream_ GUARDED_BY(mu_);
      int32 index_ GUARDED_BY(mu_) = 0;
      int32 count_ GUARDED_BY(mu_) = 0;
      int32 rows_ GUARDED_BY(mu_) = 0;
      int32 cols_ GUARDED_BY(mu_) = 0;
    };
    class ImageIterator : public Iterator {
     public:
      explicit ImageIterator(const Params& params)
          : Iterator(params) {}
      Status ReadHeader() override {
        string header;
        TF_RETURN_IF_ERROR(buffered_input_stream_->ReadNBytes(16, &header));
        if (header[0] != 0x00 || header[1] != 0x00 || header[2] != 0x08 || header[3] != 0x03) {
          return errors::InvalidArgument("mnist image file header must starts with `0x00000803`");
        }
        index_ = 0;
        count_ = (((int32)header[4] & 0xFF) << 24) | (((int32)header[5] & 0xFF) << 16) | (((int32)header[6] & 0xFF) << 8) | (((int32)header[7] & 0xFF));
        rows_ = (((int32)header[8] & 0xFF) << 24) | (((int32)header[9] & 0xFF) << 16) | (((int32)header[10] & 0xFF) << 8) | (((int32)header[11] & 0xFF));
        cols_ = (((int32)header[12] & 0xFF) << 24) | (((int32)header[13] & 0xFF) << 16) | (((int32)header[14] & 0xFF) << 8) | (((int32)header[15] & 0xFF));
        return Status::OK();
      }
      Status ReadRecord(IteratorContext* ctx, std::vector<Tensor>* out_tensors) override {
        string record;
        TF_RETURN_IF_ERROR(buffered_input_stream_->ReadNBytes(rows_ * cols_, &record));
        Tensor value_tensor(ctx->allocator({}), DT_UINT8, {rows_, cols_});
        //Tensor image_tensor(ctx->allocator({}), DT_UINT8, image_shape);
        memcpy(value_tensor.flat<uint8>().data(), record.data(), rows_ * cols_);
        out_tensors->emplace_back(std::move(value_tensor));
        return Status::OK();
      }
    };
    class LabelIterator : public Iterator {
     public:
      explicit LabelIterator(const Params& params)
          : Iterator(params) {}
      Status ReadHeader() override {
        string header;
        TF_RETURN_IF_ERROR(buffered_input_stream_->ReadNBytes(8, &header));
        if (header[0] != 0x00 || header[1] != 0x00 || header[2] != 0x08 || header[3] != 0x01) {
          return errors::InvalidArgument("mnist label file header must starts with `0x00000801`");
        }
        index_ = 0;
        count_ = (((int32)header[4] & 0xFF) << 24) | (((int32)header[5] & 0xFF) << 16) | (((int32)header[6] & 0xFF) << 8) | (((int32)header[7] & 0xFF));

        return Status::OK();
      }
      Status ReadRecord(IteratorContext* ctx, std::vector<Tensor>* out_tensors) override {
        string record;
        TF_RETURN_IF_ERROR(buffered_input_stream_->ReadNBytes(1, &record));
        Tensor value_tensor(ctx->allocator({}), DT_UINT8, {});
        memcpy(value_tensor.flat<uint8>().data(), record.data(), 1);
        out_tensors->emplace_back(std::move(value_tensor));
        return Status::OK();
      }
    };

    const std::vector<string> filenames_;
    const std::string compression_type_;
    const std::string op_;
  };
  std::string op_;
};

REGISTER_KERNEL_BUILDER(Name("MNISTImageDataset").Device(DEVICE_CPU),
                        MNISTDatasetOp);
REGISTER_KERNEL_BUILDER(Name("MNISTLabelDataset").Device(DEVICE_CPU),
                        MNISTDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
