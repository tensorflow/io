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

#include "kernels/dataset_ops.h"

#include "webp/encode.h"
#include "imageio/webpdec.h"
#include "imageio/metadata.h"
extern "C" {
#include "tiff.h"
#include "tiffio.h"
}
#include "tiffio.hxx"
#include <sstream>

namespace tensorflow {
namespace data {

class ImageStream {
 public:
  explicit ImageStream()
    : eof_(false)
    , tiff_(nullptr, TIFFClose) {
  }
  explicit ImageStream(io::InputStreamInterface& in, const string& header, const Status &status)
    : eof_(false)
    , tiff_(nullptr, TIFFClose)
    , stream_(header, std::ios_base::ate | std::ios_base::in | std::ios_base::out) {
      Status s;
      do {
        string buffer;
        s = in.ReadNBytes(4096, &buffer);
        if (s.ok() || errors::IsOutOfRange(s)) {
          stream_ << buffer;
        }
      } while (s.ok());
      tiff_.reset(TIFFStreamOpen("[in memory]", static_cast<std::istream*>(&stream_)));
  }
  int64 eof_ = false;
  std::unique_ptr<TIFF, decltype(&TIFFClose)> tiff_;
 private:
  std::stringstream stream_;
};

class ImageInput: public DataInput<ImageStream> {
 public:
  Status ReadRecord(io::InputStreamInterface& s, IteratorContext* ctx, std::unique_ptr<ImageStream>& state, int64* returned, std::vector<Tensor>* out_tensors) const override {
    if (format_ == "webp") {
      if (state.get() == nullptr) {
        state.reset(new ImageStream());
      }
      if (state->eof_) {
        *returned = 0;
        return Status::OK();
      }
      string buffer;
      TF_RETURN_IF_ERROR(s.ReadNBytes(filesize_, &buffer));

      int64 height = shape_[0];
      int64 width = shape_[1];
      int64 channel = shape_[2];

      WebPDecoderConfig config;
      WebPInitDecoderConfig(&config);
      int r = WebPGetFeatures(reinterpret_cast<const uint8_t *>(buffer.data()), buffer.size(), &config.input);
      if (r != VP8_STATUS_OK) {
        return errors::InvalidArgument("file could not be featured as WebP: ", r);
      }

      if (height != config.input.height || width != config.input.width) {
        return errors::InvalidArgument("height and width (", config.input.height, ", ", config.input.width, ") does not match data before (", height, ", ", width, ")");
      }

      *returned = 1;
      state->eof_ = true;

      Tensor value_tensor(ctx->allocator({}), DT_UINT8, {height, width, channel});

      config.output.colorspace = MODE_RGBA;
      config.output.u.RGBA.rgba = value_tensor.flat<uint8>().data();
      config.output.u.RGBA.stride = width * channel;
      config.output.u.RGBA.size = height * width * channel;
      config.output.is_external_memory = 1;
      r = WebPDecode(reinterpret_cast<const uint8_t *>(buffer.data()), buffer.size(), &config);
      if (r != VP8_STATUS_OK) {
        return errors::InvalidArgument("file could not be decoded as WebP: ", r);
      }

      out_tensors->emplace_back(std::move(value_tensor));
      return Status::OK();
    }
    else if (format_ == "tiff") {
      if (state.get() == nullptr) {
        state.reset(new ImageStream(s, "", Status::OK()));
      }
      if (state->eof_) {
        *returned = 0;
        return Status::OK();
      }

      int64 height = shape_[0];
      int64 width = shape_[1];
      int64 channel = shape_[2];

      Tensor value_tensor(ctx->allocator({}), DT_UINT8, {height, width, channel});
      // Tensor is aligned
      uint32* raster = reinterpret_cast<uint32*>(value_tensor.flat<uint8>().data());
      if (!TIFFReadRGBAImageOriented(state->tiff_.get(), width, height, raster, ORIENTATION_TOPLEFT, 0)) {
        return errors::InvalidArgument("unable to process tiff");
      }
      out_tensors->emplace_back(std::move(value_tensor));

      *returned = 1;
      if (!TIFFReadDirectory(state->tiff_.get())) {
        state->eof_ = true;
      }
      return Status::OK();
    }
    return errors::Unimplemented("format ", format_, "has not been supported yet");
  }
  Status FromStream(io::InputStreamInterface& s) override {
    string header;
    Status status = s.ReadNBytes(4096, &header);
    if (!(status.ok() || errors::IsOutOfRange(status))) {
      return status;
    }
    if (header.size() >= 12 && memcmp(&header.data()[0], "RIFF", 4) == 0 && memcmp(&header.data()[8], "WEBP", 4) == 0) {
      // 4k should be enough to capture WebP header... If not we will adjust later
      WebPDecoderConfig config;
      WebPInitDecoderConfig(&config);
      int returned = WebPGetFeatures(reinterpret_cast<const uint8_t *>(header.data()), header.size(), &config.input);
      if (returned != VP8_STATUS_OK) {
        return errors::InvalidArgument("file could not be decoded from stream as WebP: ", returned);
      }
      // Note: Always decode with channel = 4.
      int32 height = config.input.height;
      int32 width = config.input.width;
      static const int32 channel = 4;
      // Skip to the end to find out the size of WebP as we need it in the next run.
      Status status = s.SkipNBytes(std::numeric_limits<int64>::max());
      if (!(status.ok() || errors::IsOutOfRange(status))) {
        return status;
      }

      shape_ = absl::InlinedVector<int64, 3>({height, width, channel});
      filesize_ = s.Tell();
      format_ = "webp";

      return Status::OK();
    } else if (header.size() >= 4 && memcmp(&header.data()[0], "II*\0", 4) == 0) {
      // Read everything.
      ImageStream is(s, header, status);
      if (is.tiff_.get() == nullptr) {
        return errors::InvalidArgument("unable to open file");
      }
      uint32 width, height;
      TIFFGetField(is.tiff_.get(), TIFFTAG_IMAGEWIDTH, &width);
      TIFFGetField(is.tiff_.get(), TIFFTAG_IMAGELENGTH, &height);
      static int64 channel = 4;
      shape_ = absl::InlinedVector<int64, 3>({height, width, channel});
      filesize_ = s.Tell();
      format_ = "tiff";

      return Status::OK();
    }
    return errors::InvalidArgument("unknown image file format");
  }
  void EncodeAttributes(VariantTensorData* data) const override {
    data->tensors_.emplace_back(Tensor(DT_INT64, TensorShape({3})));
    data->tensors_.back().flat<int64>()(0) = shape_[0];
    data->tensors_.back().flat<int64>()(1) = shape_[1];
    data->tensors_.back().flat<int64>()(2) = shape_[2];

    data->tensors_.emplace_back(Tensor(DT_INT64, TensorShape({})));
    data->tensors_.back().scalar<int64>()() = filesize_;

    data->tensors_.emplace_back(Tensor(DT_STRING, TensorShape({})));
    data->tensors_.back().scalar<string>()() = format_;
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    size_t format_index = data.tensors().size() - 1;
    format_ = data.tensors(format_index).scalar<string>()();

    size_t filesize_index = data.tensors().size() - 2;
    filesize_ = data.tensors(filesize_index).scalar<int64>()();

    size_t shape_index = data.tensors().size() - 3;
    shape_ = absl::InlinedVector<int64, 3>({
      data.tensors(shape_index).flat<int64>()(0),
      data.tensors(shape_index).flat<int64>()(1),
      data.tensors(shape_index).flat<int64>()(2),
    });

    return true;
  }
  const string& format() const {
    return format_;
  }
 protected:
  absl::InlinedVector<int64, 3> shape_;
  int64 filesize_;
  string format_;
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(ImageInput, "tensorflow::data::ImageInput");

REGISTER_KERNEL_BUILDER(Name("ImageInput").Device(DEVICE_CPU),
                        DataInputOp<ImageInput>);
REGISTER_KERNEL_BUILDER(Name("ImageDataset").Device(DEVICE_CPU),
                        InputDatasetOp<ImageInput, ImageStream>);

}  // namespace data
}  // namespace tensorflow
