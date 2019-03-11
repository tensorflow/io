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
#include "kernels/image_ops.h"

namespace tensorflow {
namespace data {
extern "C" {
  static tmsize_t tiffclient_read(thandle_t handle, void* buf, tmsize_t tsize);
  static tmsize_t tiffclient_write(thandle_t handle, void* buf, tmsize_t tsize);
  static toff_t tiffclient_seek(thandle_t handle, toff_t toffset, int whence);
  static int tiffclient_close(thandle_t handle);
  static toff_t tiffclient_size(thandle_t handle);
  static int tiffclient_map(thandle_t handle, void** base, toff_t* tsize);
  static void tiffclient_unmap(thandle_t handle, void* base, toff_t tsize);

static tmsize_t
tiffclient_read(thandle_t handle, void* buf, tmsize_t tsize) {
  TiffFileBase *bobj = reinterpret_cast<TiffFileBase *>(handle);
  size_t size = static_cast<size_t>(tsize);
  if (static_cast<tmsize_t>(size) != tsize)
    return static_cast<tmsize_t>(-1);
  size_t result = bobj->TiffClientRead(buf, size);
  return static_cast<tmsize_t>(result);
}

static tmsize_t
tiffclient_write(thandle_t handle, void* buf, tmsize_t tsize) {
  TiffFileBase *bobj = reinterpret_cast<TiffFileBase *>(handle);
  size_t size = static_cast<size_t>(tsize);
  if (static_cast<tmsize_t>(size) != tsize)
    return static_cast<tmsize_t>(-1);
  size_t result =  bobj->TiffClientWrite(buf, size);
  return static_cast<tmsize_t>(result);
}

static toff_t
tiffclient_seek(thandle_t handle, toff_t toffset, int whence) {
  TiffFileBase *bobj = reinterpret_cast<TiffFileBase *>(handle);
  off_t offset = static_cast<off_t>(toffset);
  off_t result = bobj->TiffClientSeek(offset, whence);
  return static_cast<toff_t>(result);
}

static int
tiffclient_close(thandle_t handle) {
  TiffFileBase *bobj = reinterpret_cast<TiffFileBase *>(handle);
  return bobj->TiffClientClose();
}

static toff_t
tiffclient_size(thandle_t handle) {
  TiffFileBase *bobj = reinterpret_cast<TiffFileBase *>(handle);
  off_t result = bobj->TiffClientSize();
  return static_cast<toff_t>(result);
}

static int
tiffclient_map(thandle_t handle, void** base, toff_t* tsize) {
  TiffFileBase *bobj = reinterpret_cast<TiffFileBase *>(handle);
  off_t size;
  int result = bobj->TiffClientMap(base, &size);
  *tsize = static_cast<toff_t>(size);
  return result;
}

static void
tiffclient_unmap(thandle_t handle, void* base, toff_t tsize) {
  TiffFileBase *bobj = reinterpret_cast<TiffFileBase *>(handle);
  off_t size = static_cast<off_t>(tsize);
  return bobj->TiffClientUnmap(base, size);
}

}  //  extern "C"

Status TiffFileBase::ClientOpen(const char *name, const char* mode) {
  auto tif = TIFFClientOpen(name, mode, this,
                        tiffclient_read,
                        tiffclient_write,
                        tiffclient_seek,
                        tiffclient_close,
                        tiffclient_size,
                        tiffclient_map,
                        tiffclient_unmap);

  if (tif == NULL) {
    return errors::InvalidArgument("unable to open file:", name);
  }

  tif_.reset(tif);
  return Status::OK();
}

size_t TiffRandomFile::TiffClientRead(void* data, size_t n) {
  StringPiece result;
  Status s;
  s = file_.get()->Read(offset_, n, &result, reinterpret_cast<char *>(data));
  if (result.data() != data) {
    memmove(data, result.data(), result.size());
  }
  if (s.ok() || errors::IsOutOfRange(s)) {
    offset_ += result.size();
  }
  return  result.size();
}

off_t TiffRandomFile::TiffClientSeek(off_t offset, int whence) {
  switch (whence) {
    case SEEK_SET:
      offset_ = offset;
      break;
    case SEEK_CUR:
      offset_ += offset;
      break;
    case SEEK_END:
      offset_ = fileSize_ + offset;
      break;
    default:
      break;
  }
  return offset_;
}

off_t TiffRandomFile::TiffClientSize() {
  return fileSize_;
}

int TiffRandomFile::TiffClientClose() {
  return 0;
}


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
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          // We are currently processing a file, so try to read the next record.
          if (file_.IsOpen()) {
            unsigned int width, height;
            // get the size of the tiff
            TIFFGetField(file_.Tiff(), TIFFTAG_IMAGEWIDTH, &width);
            TIFFGetField(file_.Tiff(), TIFFTAG_IMAGELENGTH, &height);
            // RGBA
            static const int channel = 4;
            Tensor value_tensor(ctx->allocator({}), DT_UINT8, {height, width, channel});
            // Tensor is aligned
            uint32* raster = reinterpret_cast<uint32*>(value_tensor.flat<uint8_t>().data());
            if (!TIFFReadRGBAImageOriented(file_.Tiff(), width, height, raster, ORIENTATION_TOPLEFT, 0)) {
              return errors::InvalidArgument("unable to read file: ", dataset()->filenames_[current_file_index_]);
            }
            out_tensors->emplace_back(std::move(value_tensor));
            if (!TIFFReadDirectory(file_.Tiff())) {
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
        Status s = file_.Open(env, filename);
        return s;
      }

      // Resets all TIFF streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        file_.Close();
      }

      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      TiffRandomFile file_ GUARDED_BY(mu_);
    };

    const std::vector<string> filenames_;
    const DataTypeVector output_types_;
  };
  DataTypeVector output_types_;
};

REGISTER_KERNEL_BUILDER(Name("TIFFDataset").Device(DEVICE_CPU),
                        TIFFDatasetOp);

}  // namespace data
}  // namespace tensorflow
