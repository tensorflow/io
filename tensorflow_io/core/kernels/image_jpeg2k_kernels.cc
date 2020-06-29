/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "openjpeg.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace io {
namespace {

class OpjMsgCallback {
 public:
  OpjMsgCallback() {}
  ~OpjMsgCallback() {}

  string error_;

  static void InfoCallback(const char* msg, void* client_data) {
    // LOG(INFO) << "[DecodeJPEG2K]: " << msg;
  }
  static void WarningCallback(const char* msg, void* client_data) {
    LOG(WARNING) << "[DecodeJPEG2K]: " << msg;
  }
  static void ErrorCallback(const char* msg, void* client_data) {
    LOG(ERROR) << "[DecodeJPEG2K]: " << msg;
    OpjMsgCallback* p = static_cast<OpjMsgCallback*>(client_data);
    p->error_ = msg;
  }
};

class OpjStreamCallback {
 public:
  OpjStreamCallback(void* buffer, int64 length)
      : buffer_(buffer), length_(length), offset_(0) {}
  virtual ~OpjStreamCallback() {}
  void* buffer_;
  int64 length_;
  int64 offset_;

  static OPJ_SIZE_T ReadFn(void* p_buffer, OPJ_SIZE_T p_nb_bytes,
                           void* p_user_data) {
    OpjStreamCallback* p = static_cast<OpjStreamCallback*>(p_user_data);

    OPJ_SIZE_T l_nb_bytes = (p->offset_ + p_nb_bytes < p->length_)
                                ? (p_nb_bytes)
                                : (p->length_ - p->offset_);
    if (l_nb_bytes > 0) {
      memcpy(p_buffer, p->buffer_, l_nb_bytes);
    }
    p->offset_ += l_nb_bytes;

    return l_nb_bytes ? l_nb_bytes : -1;
  }
  static OPJ_OFF_T SkipFn(OPJ_OFF_T p_nb_bytes, void* p_user_data) {
    OpjStreamCallback* p = static_cast<OpjStreamCallback*>(p_user_data);

    if (p->offset_ + p_nb_bytes < p->length_) {
      p->offset_ += p_nb_bytes;
      return p_nb_bytes;
    }
    p->offset_ = p->length_;
    return (OPJ_OFF_T)-1;
  }

  static OPJ_BOOL SeekFn(OPJ_OFF_T p_nb_bytes, void* p_user_data) {
    OpjStreamCallback* p = static_cast<OpjStreamCallback*>(p_user_data);
    if (p_nb_bytes < p->length_) {
      p->offset_ = p_nb_bytes;
      return OPJ_TRUE;
    }
    p->offset_ = p->length_;
    return OPJ_FALSE;
  }

  static void OpjStreamFreeUserDataFn(void* p_user_data) {}
};

class DecodeJPEG2K : public OpKernel {
 public:
  explicit DecodeJPEG2K(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& contents_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents_tensor.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents_tensor.shape().DebugString()));
    auto contents = contents_tensor.scalar<tstring>()();

    OPJ_CODEC_FORMAT format = OPJ_CODEC_JP2;

    std::unique_ptr<opj_image_t, void (*)(opj_image_t*)> l_image(
        nullptr, [](opj_image_t* p) {
          if (p != nullptr) {
            opj_image_destroy(p);
          }
        });
    std::unique_ptr<opj_codec_t, void (*)(opj_codec_t*)> l_codec(
        opj_create_decompress(format), [](opj_codec_t* p) {
          if (p != nullptr) {
            opj_destroy_codec(p);
          }
        });

    OpjMsgCallback msg;
    opj_set_info_handler(l_codec.get(), OpjMsgCallback::InfoCallback, &msg);
    opj_set_warning_handler(l_codec.get(), OpjMsgCallback::WarningCallback,
                            &msg);
    opj_set_error_handler(l_codec.get(), OpjMsgCallback::ErrorCallback, &msg);

    std::unique_ptr<opj_stream_t, void (*)(opj_stream_t*)> l_stream(
        opj_stream_default_create(OPJ_TRUE), [](opj_stream_t* p) {
          if (p != nullptr) {
            opj_stream_destroy(p);
          }
        });
    OP_REQUIRES(context, (l_stream.get() != nullptr),
                errors::InvalidArgument("unable to create stream"));

    OpjStreamCallback data(contents.data(), contents.size());

    opj_stream_set_user_data(l_stream.get(), &data,
                             OpjStreamCallback::OpjStreamFreeUserDataFn);
    opj_stream_set_user_data_length(l_stream.get(), contents.size());
    opj_stream_set_read_function(l_stream.get(), OpjStreamCallback::ReadFn);
    opj_stream_set_skip_function(l_stream.get(), OpjStreamCallback::SkipFn);
    opj_stream_set_seek_function(l_stream.get(), OpjStreamCallback::SeekFn);

    opj_dparameters_t l_param;
    opj_set_default_decoder_parameters(&l_param);

    // TODO: adjust additional parameter with:
    // do not use layer decoding limitations
    // l_param.cp_layer = 0;
    // do not use resolutions reductions
    // l_param.cp_reduce = 0;

    OPJ_BOOL status;
    status = opj_setup_decoder(l_codec.get(), &l_param);
    OP_REQUIRES(
        context, (status),
        errors::InvalidArgument("unable to setup decoder: ", msg.error_));

    opj_image_t* p_image = nullptr;
    status = opj_read_header(l_stream.get(), l_codec.get(), &p_image);
    OP_REQUIRES(context, (status),
                errors::InvalidArgument("unable to read header: ", msg.error_));
    l_image.reset(p_image);

    OP_REQUIRES(context, ((p_image->numcomps * p_image->x1 * p_image->y1) != 0),
                errors::InvalidArgument("invalid raw image parameters"));

    for (int i = 0; i < p_image->numcomps; i++) {
      OP_REQUIRES(context, (p_image->comps[i].prec == 8),
                  errors::InvalidArgument(
                      "only 8 bit images supported, received image[", i,
                      "] = ", p_image->comps[i].prec));
    }
    switch (p_image->numcomps) {
      case 3:
      case 4:
        break;
      default:
        OP_REQUIRES(
            context, false,
            errors::InvalidArgument("only images with 3 or 4 channels are "
                                    "currently supported, received ",
                                    p_image->numcomps));
    }

    OP_REQUIRES(context, (p_image->x1 != 0 && p_image->y1 != 0),
                errors::InvalidArgument(
                    "image grid (x1, y1) cannot be zero, received (",
                    p_image->x1, ", ", p_image->y1, ")"));

    for (int i = 0; i < p_image->numcomps; i++) {
      OP_REQUIRES(
          context,
          (p_image->comps[i].w == p_image->x1) &&
              (p_image->comps[i].h == p_image->y1),
          errors::InvalidArgument("channel (", i, ") does not match image: ",
                                  p_image->comps[i].h, "x", p_image->comps[i].w,
                                  " vs. ", p_image->y1, "x", p_image->x1));
    }
    int64 width = p_image->x1;
    int64 height = p_image->y1;
    int64 channels = p_image->numcomps;

    long signed_offsets[4] = {0, 0, 0, 0};
    for (int i = 0; i < p_image->numcomps; i++) {
      if (p_image->comps[i].sgnd) {
        signed_offsets[i] = 1 << (p_image->comps[i].prec - 1);
      }
    }

    status = opj_decode(l_codec.get(), l_stream.get(), p_image);
    OP_REQUIRES(
        context, (status),
        errors::InvalidArgument("unable to decode_image: ", msg.error_));

    Tensor* image_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(
                     0, TensorShape({height, width, channels}), &image_tensor));
    auto image = image_tensor->shaped<uint8, 3>({height, width, channels});

    for (int64 i = 0; i < height; i++) {
      for (int64 j = 0; j < width; j++) {
        for (int64 k = 0; k < channels; k++) {
          uint8 value =
              p_image->comps[k].data[i * width + j] + signed_offsets[k];
          image(i, j, k) = value;
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("IO>DecodeJPEG2K").Device(DEVICE_CPU),
                        DecodeJPEG2K);

}  // namespace
}  // namespace io
}  // namespace tensorflow
