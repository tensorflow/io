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

#include "avif/avif.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace io {
namespace {

class DecodeAVIFOp : public OpKernel {
 public:
  explicit DecodeAVIFOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& contents_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents_tensor.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents_tensor.shape().DebugString()));
    auto contents = contents_tensor.scalar<tstring>()();

    avifROData raw;
    raw.data = (const uint8_t*)&contents[0];
    raw.size = contents.size();

    std::unique_ptr<avifImage, void (*)(avifImage*)> image(
        avifImageCreateEmpty(), [](avifImage* p) {
          if (p != nullptr) {
            avifImageDestroy(p);
          }
        });
    std::unique_ptr<avifDecoder, void (*)(avifDecoder*)> decoder(
        avifDecoderCreate(), [](avifDecoder* p) {
          if (p != nullptr) {
            avifDecoderDestroy(p);
          }
        });

    avifResult decodeResult = avifDecoderRead(decoder.get(), image.get(), &raw);
    OP_REQUIRES(context, (decodeResult == AVIF_RESULT_OK),
                errors::InvalidArgument("unable to decode avif: ",
                                        avifResultToString(decodeResult)));

    OP_REQUIRES(
        context, (image->depth == 8),
        errors::InvalidArgument("only 8-bit avif images are supported"));

    int64 channels = 3;

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({image->height, image->width, channels}),
                       &output_tensor));
    avifRGBImage rgb;
    avifRGBImageSetDefaults(&rgb, image.get());

    rgb.format = AVIF_RGB_FORMAT_RGB;
    rgb.depth = image->depth;

    rgb.pixels = output_tensor->flat<uint8>().data();
    rgb.rowBytes = (image->width * channels);
    avifResult rgbResult = avifImageYUVToRGB(image.get(), &rgb);
    OP_REQUIRES(context, (rgbResult == AVIF_RESULT_OK),
                errors::InvalidArgument("unable to convert avif to rgb: ",
                                        avifResultToString(rgbResult)));
  }
};
REGISTER_KERNEL_BUILDER(Name("IO>DecodeAVIF").Device(DEVICE_CPU), DecodeAVIFOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
