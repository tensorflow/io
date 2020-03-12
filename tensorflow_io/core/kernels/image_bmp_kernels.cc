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

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace io {
namespace {

class EncodeBmpOp : public OpKernel {
 public:
  explicit EncodeBmpOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    const uint32 height = input_tensor->shape().dim_size(0);
    const uint32 width = input_tensor->shape().dim_size(1);
    const uint32 channels = input_tensor->shape().dim_size(2);

    const uint32 bytes_per_px = channels;
    const uint32 line_size = bytes_per_px * width;
    const uint32 bmp_stride = (line_size + 3) & ~3;  // pad to 4
    const uint32 total_size = bmp_stride * height + bmp_header_size;

    string buffer;
    buffer.resize(total_size);
    uint8* bmp_header = (uint8*)(&buffer[0]);
    memset(bmp_header, 0x00, bmp_header_size);

    // bitmap file header
    PutLE16(bmp_header + 0, 0x4d42);            // signature 'BM'
    PutLE32(bmp_header + 2, total_size);        // size including header
    PutLE32(bmp_header + 6, 0);                 // reserved
    PutLE32(bmp_header + 10, bmp_header_size);  // offset to pixel array
    // bitmap info header
    PutLE32(bmp_header + 14, 40);                // DIB header size
    PutLE32(bmp_header + 18, width);             // dimensions
    PutLE32(bmp_header + 22, -(int)height);      // vertical flip!
    PutLE16(bmp_header + 26, 1);                 // number of planes
    PutLE16(bmp_header + 28, bytes_per_px * 8);  // bits per pixel
    PutLE32(bmp_header + 30, 0);                 // no compression (BI_RGB)
    PutLE32(bmp_header + 34, 0);                 // image size (dummy)
    PutLE32(bmp_header + 38, 2400);              // x pixels/meter
    PutLE32(bmp_header + 42, 2400);              // y pixels/meter
    PutLE32(bmp_header + 46, 0);                 // number of palette colors
    PutLE32(bmp_header + 50, 0);                 // important color count

    uint32 offset = bmp_header_size;

    // write pixel array
    for (uint32 i = 0; i < height; i++) {
      uint8* line = (uint8*)(&buffer[offset]);
      for (uint32 j = 0; j < width; j++) {
        uint8* pixel = line + j * channels;
        const uint8* data = &input_tensor->flat<uint8>()
                                 .data()[i * width * channels + j * channels];
        switch (channels) {
          case 3:
            // RGB => BGR
            pixel[0] = data[2];
            pixel[1] = data[1];
            pixel[2] = data[0];
            break;
          default:
            OP_REQUIRES(context, false,
                        errors::InvalidArgument(
                            "unsupported number of channels: ", channels));
        }
      }
      // write padding zeroes
      if (bmp_stride != line_size) {
        memset(line + line_size, 0x00, bmp_stride - line_size);
      }
      offset += bmp_stride;
    }

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &output_tensor));
    output_tensor->scalar<tstring>()() = buffer;
  }

 private:
  void PutLE16(uint8* const dst, uint32 value) {
    dst[0] = (value >> 0) & 0xff;
    dst[1] = (value >> 8) & 0xff;
  };

  void PutLE32(uint8* const dst, uint32 value) {
    PutLE16(dst + 0, (value >> 0) & 0xffff);
    PutLE16(dst + 2, (value >> 16) & 0xffff);
  };
  static const size_t bmp_header_size = 54;
};
REGISTER_KERNEL_BUILDER(Name("IO>EncodeBmp").Device(DEVICE_CPU), EncodeBmpOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
