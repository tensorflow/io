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
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/file_system.h"
#include "webp/encode.h"
#include "imageio/webpdec.h"
#include "imageio/metadata.h"

namespace tensorflow {
namespace data {
namespace {

class WebPDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;
  explicit WebPDatasetOp(OpKernelConstruction* ctx)
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
          new Iterator({this, strings::StrCat(prefix, "::WebP")}));
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
      return "WebPDatasetOp::Dataset";
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
        if (current_file_index_ == 0) {
          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        }
        if (current_file_index_ < dataset()->filenames_.size()) {
          const string& filename = dataset()->filenames_[current_file_index_];
          uint64 size = 0;
          TF_RETURN_IF_ERROR(ctx->env()->GetFileSize(filename, &size));
          std::unique_ptr<RandomAccessFile> file;
          TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(filename, &file));
          std::unique_ptr<io::RandomAccessInputStream> stream(new io::RandomAccessInputStream(file.get()));
          string data;
          TF_RETURN_IF_ERROR(stream->ReadNBytes(size, &data));
          WebPDecoderConfig config;
          WebPInitDecoderConfig(&config);
          int returned = WebPGetFeatures(reinterpret_cast<const uint8_t *>(data.c_str()),
                                         size, &config.input);
          if (returned != VP8_STATUS_OK) {
            return errors::InvalidArgument("File could not be decoded as WebP: ", filename);
          }
          // TODO (yongtang): Set channel = 4 for now.
          static const int channel = 4;
          Tensor value_tensor(ctx->allocator({}), DT_UINT8, {config.input.height, config.input.width, channel});

          config.output.colorspace = MODE_RGBA;
          config.output.u.RGBA.rgba = value_tensor.flat<uint8_t>().data();
          config.output.u.RGBA.stride = config.input.width * channel;
          config.output.u.RGBA.size = config.input.height * config.input.width * channel;
          config.output.is_external_memory = 1;
          returned = WebPDecode(reinterpret_cast<const uint8_t *>(data.c_str()), size, &config);
          if (returned != VP8_STATUS_OK) {
            return errors::InvalidArgument("File could not be decoded as WebP: ", filename);
          }
          out_tensors->emplace_back(std::move(value_tensor));
          *end_of_sequence = false;
          ++current_file_index_;
          return Status::OK();
        }
        *end_of_sequence = true;
        return Status::OK();
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
      // Sets up WebP streams to read from the topic at
      // `current_file_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_file_index_ >= dataset()->filenames_.size()) {
          return errors::InvalidArgument(
              "current_file_index_:", current_file_index_,
              " >= filenames_.size():", dataset()->filenames_.size());
        }
        return Status::OK();
      }

      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
    };

    const std::vector<string> filenames_;
    const DataTypeVector output_types_;
  };
  DataTypeVector output_types_;
};

class DecodeWebPOp : public OpKernel {
 public:
  explicit DecodeWebPOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& contents_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents_tensor.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents_tensor.shape().DebugString()));
    const auto contents = contents_tensor.scalar<string>()();

    WebPDecoderConfig config;
    WebPInitDecoderConfig(&config);
    int returned = WebPGetFeatures(reinterpret_cast<const uint8_t *>(contents.data()), contents.size(), &config.input);
    OP_REQUIRES(context, returned == VP8_STATUS_OK,
                errors::InvalidArgument("contents could not be decoded as WebP: ", returned));

    int height = config.input.height;
    int width = config.input.width;

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({height, width, channels_}), &output_tensor));

    config.output.colorspace = MODE_RGBA;
    config.output.u.RGBA.rgba = output_tensor->flat<uint8_t>().data();
    config.output.u.RGBA.stride = width * channels_;
    config.output.u.RGBA.size = height * width * channels_;
    config.output.is_external_memory = 1;

    returned = DecodeWebP(reinterpret_cast<const uint8_t *>(contents.data()), contents.size(), &config);
    OP_REQUIRES(context, returned == 0,
                errors::InvalidArgument("contents could not be decoded as WebP: ", returned));
  }

 private:
  // TODO (yongtang): Set channels_ = 4 for now.
  static const int channels_ = 4;
};
REGISTER_KERNEL_BUILDER(Name("IoWebPDataset").Device(DEVICE_CPU),
                        WebPDatasetOp);

REGISTER_KERNEL_BUILDER(Name("IoDecodeWebP").Device(DEVICE_CPU),
                        DecodeWebPOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
