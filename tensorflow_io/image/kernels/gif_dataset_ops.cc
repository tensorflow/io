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
#include "tensorflow/core/platform/file_system.h"
#include "gif_lib.h"

namespace tensorflow {
namespace data {
namespace {
extern "C" {
void DGifClose(GifFileType *GifFile) {
  // TODO (yongtang): Report warning?
  int error_code = 0;
  int status = DGifCloseFile(GifFile, &error_code);
  return;
}
}
class GIFDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;
  explicit GIFDatasetOp(OpKernelConstruction* ctx)
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
      return "GIFDatasetOp::Dataset";
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
          : DatasetIterator<Dataset>(params), file_(nullptr, DGifClose) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          // We are currently processing a file, so try to read the next record.
          if (file_) {
            unsigned int width, height;
	    SavedImage* image = &file_.get()->SavedImages[index_];
	    GifImageDesc* desc = &image->ImageDesc;
	    int imgLeft = desc->Left;
	    int imgTop = desc->Top;
	    int imgRight = desc->Left + desc->Width;
	    int imgBottom = desc->Top + desc->Height;

	    imgLeft = std::max(imgLeft, 0);
      	    imgTop = std::max(imgTop, 0);
	    imgRight = std::min(imgRight, width_);
	    imgBottom = std::min(imgBottom, height_);

	    // Already checked color map before so no more check
	    ColorMapObject* color = image->ImageDesc.ColorMap ? image->ImageDesc.ColorMap : file_.get()->SColorMap;
	    for (int i = imgTop; i < imgBottom; ++i) {
              uint8* p = (uint8*)canvas_.data() + i * width_ * channel_;
	      for (int j = imgLeft; j < imgRight; ++j) {
		GifByteType color_index = image->RasterBits[(i - desc->Top) * (desc->Width) + (j - desc->Left)];
		if (color_index >= color->ColorCount) {
                  return errors::InvalidArgument("found color index ", color_index, " outside of color map range ", color->ColorCount, " for filename: ", dataset()->filenames_[current_file_index_]);
		}
		const GifColorType& gif_color = color->Colors[color_index];
	       	p[j * channel_ + 0] = gif_color.Red;
	       	p[j * channel_ + 1] = gif_color.Green;
	       	p[j * channel_ + 2] = gif_color.Blue;
	      }
	    }
	    Tensor value_tensor(ctx->allocator({}), DT_UINT8, {height_, width_, channel_});
            std::memcpy(reinterpret_cast<char*>(value_tensor.flat<uint8_t>().data()), canvas_.data(), height_ * width_ * channel_ * sizeof(uint8_t));
            out_tensors->emplace_back(std::move(value_tensor));
	    ++index_;
            if (index_ >= frame_) {
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
	// TODO (yongtang): Could be converted to use callback with streams
	// instead of directly open files to support other file systems.
	int error_code = 0;
	GifFileType* f = DGifOpenFileName(filename.c_str(), &error_code);
	if (f == NULL) {
          return errors::InvalidArgument("unable to open file ", filename, " with error: ", GifErrorString(error_code));
	}
	file_.reset(f);
	if (error_code != D_GIF_SUCCEEDED) {
          return errors::InvalidArgument("unable to open file ", filename, " with error: ", GifErrorString(error_code));
	}
	if (DGifSlurp(f) != GIF_OK) {
          return errors::InvalidArgument("failed to slurp gif file ", filename, " with error: ", GifErrorString(f->Error));
	}
        if (f->ImageCount <= 0) {
          return errors::InvalidArgument("gif file ", filename, " does not contain any image");
        }
	if (f->SColorMap == NULL) {
          // No global color map, check frames
	  for (int i = 0; i < f->ImageCount; i++) {
            if (f->SavedImages[i].ImageDesc.ColorMap == NULL) {
              return errors::InvalidArgument("gif file ", filename, " does not contain color map for frame ", i);
	    }
	  }
	}

        height_ = 0;
        width_ = 0;
	frame_ = f->ImageCount;
        for (int i = 0; i < f->ImageCount; i++) {
          SavedImage* image = &f->SavedImages[i];
	  height_ = height_ > image->ImageDesc.Height ? height_ : image->ImageDesc.Height;
	  width_ = width_ > image->ImageDesc.Width ? width_ : image->ImageDesc.Width;
        }
	canvas_.resize(height_ * width_ * channel_);

	index_ = 0;

	std::cerr << "OPENED: " << f->ImageCount << " : " << height_ << "x" << width_ << std::endl;
	return Status::OK();
      }

      // Resets all TIFF streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
	index_ = 0;
        canvas_.clear();
	width_ = 0;
	height_ = 0;
	frame_ = 0;
        file_.reset();
      }

      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<GifFileType, decltype(&DGifClose)> file_ GUARDED_BY(mu_);
      int frame_ GUARDED_BY(mu_);
      int height_ GUARDED_BY(mu_);
      int width_ GUARDED_BY(mu_);
      std::string canvas_ GUARDED_BY(mu_);
      int index_ GUARDED_BY(mu_);
      static const int channel_ = 3;
    };

    const std::vector<string> filenames_;
    const DataTypeVector output_types_;
  };
  DataTypeVector output_types_;
};

REGISTER_KERNEL_BUILDER(Name("GIFDataset").Device(DEVICE_CPU),
                        GIFDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
