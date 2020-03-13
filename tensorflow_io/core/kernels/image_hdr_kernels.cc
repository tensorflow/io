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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow_io/core/kernels/io_stream.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace tensorflow {
namespace io {
namespace {

class DecodeHDROp : public OpKernel {
 public:
  explicit DecodeHDROp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    string input = input_tensor->scalar<tstring>()();
    OP_REQUIRES(context,
                stbi_is_hdr_from_memory((const unsigned char*)input.data(),
                                        input.size()),
                errors::InvalidArgument("not a hdr file"));

    std::unique_ptr<float, void (*)(float*)> data(nullptr, [](float* p) {
      if (p != nullptr) {
        stbi_image_free(p);
      }
    });

    int desired_channels = 3;
    int x, y, channels_in_file;
    data.reset(stbi_loadf_from_memory((const unsigned char*)input.data(),
                                      input.size(), &x, &y, &channels_in_file,
                                      desired_channels));

    OP_REQUIRES(context, (data.get() != nullptr),
                errors::InvalidArgument("unable to open as a hdr file"));
    OP_REQUIRES(context, (x != 0 && y != 0 && channels_in_file == 3),
                errors::InvalidArgument("invalid shape: (", x, ", ", y, ", ",
                                        channels_in_file, ")"));

    int64 channels = static_cast<int64>(channels_in_file);
    int64 height = static_cast<int64>(y);
    int64 width = static_cast<int64>(x);

    Tensor* image_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(
                     0, TensorShape({height, width, channels}), &image_tensor));
    auto image = image_tensor->shaped<float, 3>({height, width, channels});

    // Check padding?
    memcpy(image_tensor->flat<float>().data(), data.get(),
           height * width * channels * sizeof(float));
  }

 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
REGISTER_KERNEL_BUILDER(Name("IO>DecodeHdr").Device(DEVICE_CPU), DecodeHDROp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
