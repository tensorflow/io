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

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "imageio/webpdec.h"
#include "imageio/metadata.h"
#include "kernels/image_ops.h"
namespace tensorflow {
namespace data {
class InputSequenceInterface : public ResourceBase {
 public:
  virtual ~InputSequenceInterface() override {}
  virtual Status Initialize(const std::vector<string>& names) = 0;
  virtual Status Count(int64* count) = 0;
  virtual Status Type(std::vector<string>* type) = 0;
};
class ImageInputSequence : public InputSequenceInterface {
 public:
  ImageInputSequence(Env* env)
      : InputSequenceInterface(),
        env_(env) {}

  Status Initialize(const std::vector<string>& names) override {

    std::vector<int32> indices;
    indices.clear();
    std::vector<string> types;
    types.clear();

    // First process files
    for (int i = 0; i < names.size(); ++i) {
      std::unique_ptr<RandomAccessFile> file;
      TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(names[i], &file));
      char buffer[12];
      StringPiece result;
      TF_RETURN_IF_ERROR(file->Read(0, sizeof(buffer), &result, buffer));
      if (memcmp(&buffer[0], "RIFF", 4) == 0 && memcmp(&buffer[8], "WEBP", 4) == 0) {
        // WebP
	uint64 size = 0;
	env_->GetFileSize(names[i], &size);
        string data;
	data.resize(size);
        TF_RETURN_IF_ERROR(file->Read(0, size, &result, &data[0]));
        WebPDecoderConfig config;
        WebPInitDecoderConfig(&config);
        int returned = WebPGetFeatures(reinterpret_cast<const uint8_t *>(data.c_str()), size, &config.input);
        if (returned != VP8_STATUS_OK) {
          return errors::InvalidArgument("File could not be decoded as WebP: ", names[i]);
        }
        // Note: Always decode with channel = 4.
	static const int32 channel = 4;
	int32 height = config.input.height;
	int32 width = config.input.width;

	indices.push_back(i);
	types.push_back("webp");
      } else if (memcmp(&buffer[0], "II*\0", 4) == 0) {
        // TIFF
	TiffRandomFile f;
	TF_RETURN_IF_ERROR(f.Open(env_, names[i]));
	do {
          // Note: Always decode with channel = 4.
	  static const int32 channel = 4;
	  int32 height;
          int32 width;
          // get the size of the tiff
          TIFFGetField(f.Tiff(), TIFFTAG_IMAGEWIDTH, &width);
          TIFFGetField(f.Tiff(), TIFFTAG_IMAGELENGTH, &height);
	  indices.push_back(i);
	  types.push_back("tiff");
	} while (TIFFReadDirectory(f.Tiff()));
      } else {
        return errors::InvalidArgument("unable to find out the image type for: ", names[i]);
      }
    }
    mutex_lock l(mu_);
    filenames_.resize(names.size());
    for (int i = 0; i < names.size(); ++i) {
      filenames_[i] = names[i];
    }
    indices_.resize(indices.size());
    for (int i = 0; i < indices.size(); ++i) {
      indices_[i] = indices[i];
    }
    types_.resize(types.size());
    for (int i = 0; i < types.size(); ++i) {
      types_[i] = types[i];
    }
    return Status::OK();
  }

  Status Count(int64* count) override {
    mutex_lock l(mu_);
    *count = indices_.size();
    return Status::OK();
  }

  Status Type(std::vector<string>* types) override {
    mutex_lock l(mu_);
    types->resize(types_.size());
    for (int i = 0; i < types_.size(); i++) {
      (*types)[i] = types_[i];
    }
    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("ImageInputSequence[]");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::vector<string> filenames_ GUARDED_BY(mu_);
  std::vector<int32> indices_ GUARDED_BY(mu_);
  std::vector<string> types_ GUARDED_BY(mu_);
};

class ImageInputSequenceOp : public ResourceOpKernel<InputSequenceInterface> {
 public:
  explicit ImageInputSequenceOp(OpKernelConstruction* context)
      : ResourceOpKernel<InputSequenceInterface>(context) {
    env_ = context->env();
  }
 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<InputSequenceInterface>::Compute(context);
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(context, context->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        context, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filename` must be a scalar or vector."));

    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }

    OP_REQUIRES_OK(context, resource_->Initialize(filenames));
  }
  Status CreateResource(InputSequenceInterface** sequence)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *sequence = new ImageInputSequence(env_);
    return Status::OK();
  }
  Env* env_;
};
class ImageInputSequenceCountOp : public OpKernel {
 public:
  explicit ImageInputSequenceCountOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    InputSequenceInterface* sequence;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "sequence", &sequence));
    int64 count = 0;
    Status status = sequence->Count(&count);
    sequence->Unref();
    OP_REQUIRES_OK(context, status);
    Tensor* count_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &count_tensor));
    count_tensor->scalar<int64>()() = count;
  }
};
class ImageInputSequenceTypeOp : public OpKernel {
 public:
  explicit ImageInputSequenceTypeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    InputSequenceInterface* sequence;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "sequence", &sequence));
    std::vector<string> type;
    Status status = sequence->Type(&type);
    sequence->Unref();
    OP_REQUIRES_OK(context, status);
    Tensor* type_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({type.size()}), &type_tensor));
    for (int i = 0; i < type.size(); i++) {
      type_tensor->flat<string>()(i) = type[i];
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("ImageInputSequence").Device(DEVICE_CPU),
                        ImageInputSequenceOp);
REGISTER_KERNEL_BUILDER(Name("ImageInputSequenceCount").Device(DEVICE_CPU),
                        ImageInputSequenceCountOp);
REGISTER_KERNEL_BUILDER(Name("ImageInputSequenceType").Device(DEVICE_CPU),
                        ImageInputSequenceTypeOp);

}  // namespace data
}  // namespace tensorflow
