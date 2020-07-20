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

#include "gif_lib.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace io {
namespace {

extern "C" {
int GifQuantizeBuffer(unsigned int Width, unsigned int Height,
                      int* ColorMapSize, GifByteType* RedInput,
                      GifByteType* GreenInput, GifByteType* BlueInput,
                      GifByteType* OutputBuffer, GifColorType* OutputColorMap);
}

class EncodeGifOp : public OpKernel {
 public:
  explicit EncodeGifOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    OP_REQUIRES(context, input_tensor->dims() == 4,
                errors::InvalidArgument("The rank of the input should be 4"));

    const int64 count = input_tensor->shape().dim_size(0);
    const int64 height = input_tensor->shape().dim_size(1);
    const int64 width = input_tensor->shape().dim_size(2);
    const int64 channels = input_tensor->shape().dim_size(3);
    OP_REQUIRES(
        context, (channels == 3),
        errors::InvalidArgument(
            "only rgb (channel=3) mode encoding supported: ", channels));

    // create a GIF color map object

    int color_size = 256;
    std::unique_ptr<ColorMapObject, void (*)(ColorMapObject*)> color(
        GifMakeMapObject(color_size, NULL), [](ColorMapObject* p) {
          if (p != nullptr) {
            GifFreeMapObject(p);
          }
        });
    OP_REQUIRES(context, (color.get() != nullptr),
                errors::InvalidArgument("unable to create color map"));

    std::vector<unsigned char> r_data(count * height * width);
    unsigned char* r_ptr = &r_data[0];
    std::vector<unsigned char> g_data(count * height * width);
    unsigned char* g_ptr = &g_data[0];
    std::vector<unsigned char> b_data(count * height * width);
    unsigned char* b_ptr = &b_data[0];
    std::vector<unsigned char> q_data(count * height * width);
    unsigned char* q_ptr = &q_data[0];

    for (int64 i = 0; i < count; i++) {
      for (int64 h = 0; h < height; h++) {
        for (int64 w = 0; w < width; w++) {
          r_ptr[i * height * width + h * width + w] =
              input_tensor->tensor<uint8, 4>()(i, h, w, 0);
          g_ptr[i * height * width + h * width + w] =
              input_tensor->tensor<uint8, 4>()(i, h, w, 1);
          b_ptr[i * height * width + h * width + w] =
              input_tensor->tensor<uint8, 4>()(i, h, w, 2);
        }
      }
    }

    int status;

    status = GifQuantizeBuffer(count * width, height, &color_size, r_ptr, g_ptr,
                               b_ptr, q_ptr, color->Colors);
    OP_REQUIRES(context, (status == GIF_OK),
                errors::InvalidArgument("unable to quantize buffer"));
    int error_code;

    // allocate 64k + num_pixels (1 byte/pixel)
    tstring buffer;
    buffer.reserve(64 * 1024 + count * height * width);

    std::unique_ptr<GifFileType, void (*)(GifFileType*)> file(
        EGifOpen(&buffer, GifOutputFunc, &error_code), [](GifFileType* p) {
          if (p != nullptr) {
            int error_code = 0;
            int status = EGifCloseFile(p, &error_code);
          }
        });

    OP_REQUIRES(context, (file.get() != nullptr),
                errors::InvalidArgument("unable to open gif file for write: ",
                                        GifErrorString(error_code)));

    EGifSetGifVersion(file.get(), true);

    status = EGifPutScreenDesc(file.get(), width, height, 8, 0, color.get());
    OP_REQUIRES(context, (status == GIF_OK),
                errors::InvalidArgument("unable to put screen desc: ",
                                        GifErrorString(file->Error)));

    for (int64 i = 0; i < count; i++) {
      status = EGifPutImageDesc(file.get(), 0, 0, width, height, false, NULL);
      OP_REQUIRES(context, (status == GIF_OK),
                  errors::InvalidArgument("unable to put image desc: ",
                                          GifErrorString(file->Error)));

      for (int64 h = 0; h < height; h++) {
        int64 offset = i * height * width + h * width;
        GifPixelType* p = q_ptr + offset;
        status = EGifPutLine(file.get(), p, width);
        OP_REQUIRES(context, (status == GIF_OK),
                    errors::InvalidArgument("unable to write line: ",
                                            GifErrorString(file->Error)));
      }
    }

    file.reset(nullptr);
    color.reset(nullptr);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &output_tensor));
    output_tensor->scalar<tstring>()() = buffer;
  }

  static int GifOutputFunc(GifFileType* f, const GifByteType* buffer,
                           int size) {
    tstring* p = static_cast<tstring*>(f->UserData);
    size_t offset = p->size();
    p->resize(offset + size);
    if (size > 0) {
      memcpy(&(*p)[offset], buffer, size);
    }
    return size;
  }
};
REGISTER_KERNEL_BUILDER(Name("IO>EncodeGif").Device(DEVICE_CPU), EncodeGifOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
