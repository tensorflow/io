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

#include "imageio/metadata.h"
#include "imageio/webpdec.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/file_system.h"
#include "webp/encode.h"

namespace tensorflow {
namespace io {
namespace {

class DecodeWebPOp : public OpKernel {
 public:
  explicit DecodeWebPOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& contents_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents_tensor.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents_tensor.shape().DebugString()));
    auto contents = contents_tensor.scalar<tstring>()();

    WebPDecoderConfig config;
    WebPInitDecoderConfig(&config);
    int returned =
        WebPGetFeatures(reinterpret_cast<const uint8_t*>(contents.data()),
                        contents.size(), &config.input);
    OP_REQUIRES(context, returned == VP8_STATUS_OK,
                errors::InvalidArgument(
                    "contents could not be decoded as WebP: ", returned));

    int height = config.input.height;
    int width = config.input.width;

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({height, width, channels_}),
                                &output_tensor));

    config.output.colorspace = MODE_RGBA;
    config.output.u.RGBA.rgba = output_tensor->flat<uint8_t>().data();
    config.output.u.RGBA.stride = width * channels_;
    config.output.u.RGBA.size = height * width * channels_;
    config.output.is_external_memory = 1;

    returned = DecodeWebP(reinterpret_cast<const uint8_t*>(contents.data()),
                          contents.size(), &config);
    OP_REQUIRES(context, returned == 0,
                errors::InvalidArgument(
                    "contents could not be decoded as WebP: ", returned));
  }

 private:
  // TODO (yongtang): Set channels_ = 4 for now.
  static const int channels_ = 4;
};
REGISTER_KERNEL_BUILDER(Name("IO>DecodeWebP").Device(DEVICE_CPU), DecodeWebPOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
