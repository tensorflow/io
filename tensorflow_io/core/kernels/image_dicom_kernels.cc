/* Copyright 2019 Gradient Health Inc. All Rights Reserved.
   Author: Marcelo Lerendegui <marcelo@gradienthealth.io>

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

// clang-format off
#include "dcmtk/config/osconfig.h"

#include "dcmtk/dcmdata/dcfilefo.h"

#include "dcmtk/dcmdata/dcdict.h"
#include "dcmtk/dcmdata/dcistrmb.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/ofstd/ofstdinc.h"
#include "dcmtk/ofstd/ofstring.h"
#include "dcmtk/ofstd/oftypes.h"

#include "dcmtk/dcmimgle/diutils.h"

#include "dcmtk/dcmimage/dipipng.h"  // for dcmimage PNG plugin
#include "dcmtk/dcmimage/dipitiff.h" // for dcmimage TIFF plugin
#include "dcmtk/dcmjpeg/dipijpeg.h"  // for dcmimage JPEG plugin

#include "dcmtk/dcmimage/diregist.h"
#include "dcmtk/dcmimgle/dcmimage.h"

#include "dcmtk/dcmdata/dcrledrg.h" // for DcmRLEDecoderRegistration
#include "dcmtk/dcmjpeg/djdecode.h" // for dcmjpeg decoders
#include "dcmtk/dcmjpls/djdecode.h" // for dcmjpls decoders

#include "fmjpeg2k/djdecode.h" // for fmjpeg2koj decoders

#include <cstdint>
#include <exception>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "absl/strings/str_split.h"
#include "absl/strings/numbers.h"

// clang-format on

typedef uint64_t
    Uint64;  // Uint64 not present in tensorflow::custom-op docker image dcmtk

namespace tensorflow {
namespace io {
namespace {

// FMJPEG2K is not safe to cleanup, so use DecoderRegistration
// to provide protection and only cleanup during program exit.
class DecoderRegistration {
 public:
  static void registerCodecs() { instance().registration(); }
  static void cleanup() {}

 private:
  explicit DecoderRegistration() : initialized_(false) {}
  ~DecoderRegistration() {
    mutex_lock l(mu_);
    if (initialized_) {
      DcmRLEDecoderRegistration::cleanup();    // deregister RLE codecs
      DJDecoderRegistration::cleanup();        // deregister JPEG codecs
      DJLSDecoderRegistration::cleanup();      // deregister JPEG-LS codecs
      FMJPEG2KDecoderRegistration::cleanup();  // deregister fmjpeg2koj
      initialized_ = false;
    }
  }

  void registration() {
    mutex_lock l(mu_);
    if (!initialized_) {
      DcmRLEDecoderRegistration::registerCodecs();    // register RLE codecs
      DJDecoderRegistration::registerCodecs();        // register JPEG codecs
      DJLSDecoderRegistration::registerCodecs();      // register JPEG-LS codecs
      FMJPEG2KDecoderRegistration::registerCodecs();  // register fmjpeg2koj
      initialized_ = true;
    }
  }
  static DecoderRegistration &instance() {
    static DecoderRegistration decoder_registration;
    return decoder_registration;
  }

 private:
  mutex mu_;
  bool initialized_ TF_GUARDED_BY(mu_);
};

template <typename dtype>
class DecodeDICOMImageOp : public OpKernel {
 public:
  explicit DecodeDICOMImageOp(OpKernelConstruction *context)
      : OpKernel(context) {
    // Get the on_error
    OP_REQUIRES_OK(context, context->GetAttr("on_error", &on_error_));

    // Get the scale
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));

    // Get the color_dim
    OP_REQUIRES_OK(context, context->GetAttr("color_dim", &color_dim_));

    DecoderRegistration::registerCodecs();
  }

  ~DecodeDICOMImageOp() { DecoderRegistration::cleanup(); }

  void Compute(OpKernelContext *context) override {
    // Grab the input file content tensor
    const Tensor &in_contents = context->input(0);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(in_contents.shape()),
        errors::InvalidArgument("DecodeDICOMImage expects input content tensor "
                                "to be scalar, but had shape: ",
                                in_contents.shape().DebugString()));

    const auto in_contents_scalar = in_contents.scalar<tstring>()();

    // Load Dicom Image
    DcmInputBufferStream data_buf;
    data_buf.setBuffer(in_contents_scalar.data(), in_contents_scalar.length());
    data_buf.setEos();

    DcmFileFormat dicom_file;
    dicom_file.transferInit();
    OFCondition cond = dicom_file.read(data_buf);
    dicom_file.transferEnd();

    DicomImage *image = NULL;
    try {
      image = new DicomImage(&dicom_file, EXS_Unknown,
                             CIF_DecompressCompletePixelData);
    } catch (...) {
      image = NULL;
    }

    unsigned long frameWidth = 0;
    unsigned long frameHeight = 0;
    unsigned int pixelDepth = 0;
    unsigned long dataSize = 0;
    unsigned long frameCount = 0;
    unsigned int samples_per_pixel = 0;

    if ((image == NULL) || (image->getStatus() != EIS_Normal)) {
      if (on_error_ == "strict") {
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Error loading image"));
        return;
      } else if ((on_error_ == "skip") || (on_error_ == "lossy")) {
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, {0}, &output_tensor));
        return;
      }
    }

    // Get image information
    frameCount = image->getFrameCount();  // getNumberOfFrames(); starts at
                                          // version DCMTK-3.6.1_20140617
    frameWidth = image->getWidth();
    frameHeight = image->getHeight();
    pixelDepth = image->getDepth();
    samples_per_pixel = image->isMonochrome() ? 1 : 3;

    // Create an output tensor shape
    TensorShape out_shape;
    out_shape = TensorShape({static_cast<int64>(frameCount),
                             static_cast<int64>(frameHeight),
                             static_cast<int64>(frameWidth),
                             static_cast<int64>(samples_per_pixel)});

    // Check if output type is ok for image
    if (pixelDepth > sizeof(dtype) * 8) {
      if (on_error_ == "strict") {
        OP_REQUIRES(
            context, false,
            errors::InvalidArgument(
                "Input argument dtype size smaller than pixelDepth (bits):",
                pixelDepth));
        return;
      } else if (on_error_ == "skip") {
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, {0}, &output_tensor));
        return;
      }
    }

    // Create an output tensor
    Tensor *output_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output_tensor));

    auto output_flat = output_tensor->template flat<dtype>();

    unsigned long frame_pixel_count =
        frameHeight * frameWidth * samples_per_pixel;

    for (Uint64 f = 0; f < frameCount; f++) {
      const void *image_frame = image->getOutputData(pixelDepth, f);

      for (Uint64 p = 0; p < frame_pixel_count; p++) {
        output_flat(f * frame_pixel_count + p) =
            convert_uintn_to_t(image_frame, pixelDepth, p);
      }
    }
    delete image;
  }

  dtype convert_uintn_to_t(const void *buff, unsigned int n_bits,
                           unsigned int pos) {
    uint64 in_value;
    if (n_bits <= 8)
      in_value = *((const uint8 *)buff + pos);
    else if (n_bits <= 16)
      in_value = *((const uint16 *)buff + pos);
    else if (n_bits <= 32)
      in_value = *((const uint32 *)buff + pos);
    else
      in_value = *((const uint64 *)buff + pos);

    dtype out;
    uint64_to_t(in_value, n_bits, &out);
    return out;
  }

  void uint64_to_t(uint64 in_value, unsigned int n_bits, uint8 *out_value) {
    if (scale_ == "auto") {
      in_value = in_value << (64 - n_bits);
      *out_value = (dtype)(in_value >> (64 - 8 * sizeof(uint8)));
    } else if (scale_ == "preserve") {
      *out_value = in_value >= ((1ULL << 8 * sizeof(uint8)) - 1)
                       ? (dtype)(((1ULL << 8 * sizeof(uint8)) - 1))
                       : (dtype)(in_value);
    }
  }

  void uint64_to_t(uint64 in_value, unsigned int n_bits, uint16 *out_value) {
    if (scale_ == "auto") {
      in_value = in_value << (64 - n_bits);
      *out_value = (dtype)(in_value >> (64 - 8 * sizeof(uint16)));
    } else if (scale_ == "preserve") {
      *out_value = in_value >= ((1ULL << 8 * sizeof(uint16)) - 1)
                       ? (dtype)(((1ULL << 8 * sizeof(uint16)) - 1))
                       : (dtype)(in_value);
    }
  }

  void uint64_to_t(uint64 in_value, unsigned int n_bits, uint32 *out_value) {
    if (scale_ == "auto") {
      in_value = in_value << (64 - n_bits);
      *out_value = (dtype)(in_value >> (64 - 8 * sizeof(uint32)));
    } else if (scale_ == "preserve") {
      *out_value = in_value >= ((1ULL << 8 * sizeof(uint32)) - 1)
                       ? (dtype)(((1ULL << 8 * sizeof(uint32)) - 1))
                       : (dtype)(in_value);
    }
  }

  void uint64_to_t(uint64 in_value, unsigned int n_bits, uint64 *out_value) {
    *out_value = in_value;
  }

  void uint64_to_t(uint64 in_value, unsigned int n_bits, float *out_value) {
    if (scale_ == "auto")
      *out_value = (float)(in_value) / (float)((1ULL << n_bits) - 1);
    else if (scale_ == "preserve")
      *out_value = (float)(in_value);
  }

  void uint64_to_t(uint64 in_value, unsigned int n_bits,
                   Eigen::half *out_value) {
    if (scale_ == "auto")
      *out_value = static_cast<Eigen::half>((double)(in_value) /
                                            (double)((1ULL << n_bits) - 1));
    else if (scale_ == "preserve")
      *out_value = static_cast<Eigen::half>(in_value);
  }

  void uint64_to_t(uint64 in_value, unsigned int n_bits, double *out_value) {
    if (scale_ == "auto")
      *out_value = (double)(in_value) / (double)((1ULL << n_bits) - 1);
    else if (scale_ == "preserve")
      *out_value = (double)(in_value);
  }

  string on_error_;
  string scale_;
  bool color_dim_;
};

class DecodeDICOMDataOp : public OpKernel {
 public:
  explicit DecodeDICOMDataOp(OpKernelConstruction *context)
      : OpKernel(context) {}

  ~DecodeDICOMDataOp() {}

  void Compute(OpKernelContext *context) override {
    // Grab the input file content tensor
    const Tensor &in_contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(in_contents.shape()),
                errors::InvalidArgument("DecodeDICOMData expects input content "
                                        "tensor to be scalar, but had shape: ",
                                        in_contents.shape().DebugString()));

    const auto in_contents_scalar = in_contents.scalar<tstring>()();

    const Tensor *in_tags;
    OP_REQUIRES_OK(context, context->input("tags", &in_tags));

    // Create an output tensor
    Tensor *out_tag_values = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, in_tags->shape(),
                                                     &out_tag_values));

    auto out_tag_values_flat = out_tag_values->flat<tstring>();

    DcmInputBufferStream dataBuf;
    dataBuf.setBuffer(in_contents_scalar.data(), in_contents_scalar.length());
    dataBuf.setEos();

    DcmFileFormat dfile;
    dfile.transferInit();
    OFCondition cond = dfile.read(dataBuf);
    dfile.transferEnd();

    DcmItem *item = static_cast<DcmItem *>(dfile.getDataset());
    DcmMetaInfo *meta = dfile.getMetaInfo();

    for (int64 tag_i = 0; tag_i < in_tags->NumElements(); ++tag_i) {
      DcmTag tag;
      if (in_tags->dtype() == DT_STRING) {
        uint16 tag_group_number, tag_element_number;
        const auto &in_tags_flat = in_tags->flat<tstring>();
        const tstring &tag_sequence = in_tags_flat(tag_i);
        std::vector<absl::string_view> tag_sequence_views;
        if (tag_sequence[0] == '[' &&
            tag_sequence[tag_sequence.size() - 1] == ']') {
          tag_sequence_views =
              absl::StrSplit(absl::string_view(tag_sequence.data() + 1,
                                               tag_sequence.size() - 2),
                             "][");
        } else {
          tag_sequence_views.push_back(
              absl::string_view(tag_sequence.data(), tag_sequence.size()));
        }

        OP_REQUIRES(
            context, (tag_sequence_views.size() % 2 == 1),
            errors::InvalidArgument(
                "tag sequences should have 2xn + 1 elements, received: ",
                tag_sequence_views.size()));

        // Walk through before the last element of value
        for (size_t i = 0; i < tag_sequence_views.size() - 1; i += 2) {
          absl::string_view tag_value(tag_sequence_views[i].data(),
                                      tag_sequence_views[i].size());
          OP_REQUIRES_OK(context, GetDcmTag(tag_value, &tag));

          absl::string_view number_value(tag_sequence_views[i + 1].data(),
                                         tag_sequence_views[i + 1].size());
          uint32 number = 0;
          OP_REQUIRES(
              context,
              absl::numbers_internal::safe_strtou32_base(number_value, &number,
                                                         0),
              errors::InvalidArgument("number should be an integer, received ",
                                      number_value));

          DcmItem *lookup;
          OFCondition condition =
              item->findAndGetSequenceItem(tag, lookup, number);
          OP_REQUIRES(context, condition.good(),
                      errors::InvalidArgument("item findAndGetSequenceItem: ",
                                              condition.text()));
          item = lookup;
        }

        // The last element of value
        absl::string_view tag_value(tag_sequence_views.back().data(),
                                    tag_sequence_views.back().size());
        OP_REQUIRES_OK(context, GetDcmTag(tag_value, &tag));
      } else {
        const auto &in_tags_flat = in_tags->flat<uint32>();
        uint32 tag_value = in_tags_flat(tag_i);
        OP_REQUIRES_OK(context, GetDcmTag(tag_value, &tag));
      }

      OFString val;
      if (item->tagExists(tag)) {
        OFCondition condition = item->findAndGetOFStringArray(tag, val);
        OP_REQUIRES(context, condition.good(),
                    errors::InvalidArgument("item findAndGetOFStringArray: ",
                                            condition.text()));
      } else if (meta->tagExists(tag)) {
        OFCondition condition = meta->findAndGetOFStringArray(tag, val);
        OP_REQUIRES(context, condition.good(),
                    errors::InvalidArgument("meta findAndGetOFStringArray: ",
                                            condition.text()));
      } else {
        val = OFString("");
      }
      out_tag_values_flat(tag_i) = val.c_str();
    }
  }

 private:
  Status GetDcmTag(const uint32 tag_value, DcmTag *tag) {
    uint16 tag_group_number = (uint16)((tag_value & 0xFFFF0000) >> 16);
    uint16 tag_element_number = (uint16)((tag_value & 0x0000FFFF) >> 0);
    *tag = DcmTag(tag_group_number, tag_element_number);
    return OkStatus();
  }
  Status GetDcmTag(const absl::string_view tag_value, DcmTag *tag) {
    std::vector<absl::string_view> number_views =
        absl::StrSplit(tag_value, ',');
    if (number_views.size() != 2) {
      return errors::InvalidArgument(
          "sequence should consist of group and "
          "element numbers, received ",
          tag_value);
    }
    uint32 number = 0;
    if (!absl::numbers_internal::safe_strtou32_base(number_views[0], &number,
                                                    0)) {
      return errors::InvalidArgument(
          "group number should be an integer, received ", number_views[0]);
    }
    if (number > std::numeric_limits<short>::max()) {
      return errors::InvalidArgument("group number should be uint16, received ",
                                     number_views[0]);
    }
    uint16 tag_group_number = number;

    if (!absl::numbers_internal::safe_strtou32_base(number_views[1], &number,
                                                    0)) {
      return errors::InvalidArgument(
          "element number should be an integer, received ", number_views[1]);
    }
    if (number > std::numeric_limits<short>::max()) {
      return errors::InvalidArgument(
          "element number should be uint16, received ", number_views[1]);
    }
    uint16 tag_element_number = number;

    *tag = DcmTag(tag_group_number, tag_element_number);
    return OkStatus();
  }
};

// Register the CPU kernels.
#define REGISTER_DECODE_DICOM_IMAGE_CPU(dtype)                 \
  REGISTER_KERNEL_BUILDER(Name("IO>DecodeDICOMImage")          \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<dtype>("dtype"), \
                          DecodeDICOMImageOp<dtype>);

REGISTER_DECODE_DICOM_IMAGE_CPU(uint8);
REGISTER_DECODE_DICOM_IMAGE_CPU(uint16);
REGISTER_DECODE_DICOM_IMAGE_CPU(uint32);
REGISTER_DECODE_DICOM_IMAGE_CPU(uint64);
REGISTER_DECODE_DICOM_IMAGE_CPU(float);
REGISTER_DECODE_DICOM_IMAGE_CPU(Eigen::half);
REGISTER_DECODE_DICOM_IMAGE_CPU(double);

#undef REGISTER_DECODE_DICOM_IMAGE_CPU

REGISTER_KERNEL_BUILDER(Name("IO>DecodeDICOMData").Device(DEVICE_CPU),
                        DecodeDICOMDataOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
