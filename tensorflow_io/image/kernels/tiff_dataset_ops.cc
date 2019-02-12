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
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/platform/file_system.h"
#include "tiff.h"
#include "tiffio.h"

namespace tensorflow {
namespace data {
namespace {

class TIFFDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;
  explicit TIFFDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
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

    *output = new Dataset(ctx, filenames);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const std::vector<string>& filenames)
        : DatasetBase(DatasetContext(ctx)),
          filenames_(filenames) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::TIFF")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_UINT8});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{-1, -1, -1}});
      return *shapes;
    }

    string DebugString() const override {
      return "TIFFDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {filenames}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params), file_(nullptr, TIFFClose) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          // We are currently processing a file, so try to read the next record.
          if (file_) {
            unsigned int width, height;
	    // get the size of the tiff
	    TIFFGetField(file_.get(), TIFFTAG_IMAGEWIDTH, &width);
	    TIFFGetField(file_.get(), TIFFTAG_IMAGELENGTH, &height);
	    // get the total number of pixels
	    unsigned int npixels = width*height;
	    uint32* raster = (uint32*)_TIFFmalloc(npixels * sizeof(uint32));
	    if (!TIFFReadRGBAImageOriented(file_.get(), width, height, raster, 	ORIENTATION_TOPLEFT, 0)) {
              _TIFFfree(raster);
              return errors::InvalidArgument("unable to read file: ", dataset()->filenames_[current_file_index_]);
	    }
	    // RGBA
	    static const int channel = 4;
	    Tensor value_tensor(ctx->allocator({}), DT_UINT8, {height, width, channel});
	    int num_bytes = npixels * sizeof(uint32);
            std::memcpy(reinterpret_cast<char*>(value_tensor.flat<uint8_t>().data()), raster, num_bytes * sizeof(uint8_t));
            out_tensors->emplace_back(std::move(value_tensor));
	    _TIFFfree(raster);
            if (!TIFFReadDirectory(file_.get())) {
              ResetStreamsLocked();
              ++current_file_index_;
	    }
            *end_of_sequence = false;
            return Status::OK();
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

     private:
      // Sets up TIFF streams to read from the topic at
      // `current_file_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_file_index_ >= dataset()->filenames_.size()) {
          return errors::InvalidArgument(
              "current_file_index_:", current_file_index_,
              " >= filenames_.size():", dataset()->filenames_.size());
        }

        // Actually move on to next file.
        const string& filename = dataset()->filenames_[current_file_index_];
	TIFF* f = TIFFOpen(filename.c_str(), "r");
	if (f == NULL) {
          return errors::InvalidArgument("unable to open file:", filename);
	}
	file_.reset(f);
	return Status::OK();
      }

      // Resets all TIFF streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        file_.reset();
      }

      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<TIFF, decltype(&TIFFClose)> file_ GUARDED_BY(mu_);
    };

    const std::vector<string> filenames_;
    const DataTypeVector output_types_;
  };
  DataTypeVector output_types_;
};

REGISTER_KERNEL_BUILDER(Name("TIFFDataset").Device(DEVICE_CPU),
                        TIFFDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
