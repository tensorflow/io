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

namespace tensorflow {
namespace io {
namespace {
class DecodePNMOp : public OpKernel {
 public:
  explicit DecodePNMOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<string>()();
    size_t pos = 0;
    size_t off = input.find_first_of(" \t\r\n", pos);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no magic"));
    string magic = input.substr(pos, off - pos);
    OP_REQUIRES(
        context,
        (magic == "P2" || magic == "P3" || magic == "P5" || magic == "P6"),
        errors::InvalidArgument("invalid format: ", magic));
    const int64 channels = (magic == "P2" || magic == "P5") ? 1 : 3;

    off = input.find_first_not_of(" \t\r\n", off);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no width"));
    if (input[off] == '#') {
      // comment
      while (off < input.size() && input[off] != '\n') {
        off++;
      }
    }
    // width, height, max
    pos = off;
    off = input.find_first_of(" \t\r\n", pos);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no width"));
    int64 width;
    OP_REQUIRES(context,
                (strings::safe_strto64(input.substr(pos, off - pos), &width)),
                errors::InvalidArgument("unable to parse width: ",
                                        input.substr(pos, off - pos)));

    off = input.find_first_not_of(" \t\r\n", off);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no height"));
    pos = off;
    off = input.find_first_of(" \t\r\n", pos);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no height"));
    int64 height;
    OP_REQUIRES(context,
                (strings::safe_strto64(input.substr(pos, off - pos), &height)),
                errors::InvalidArgument("unable to parse height: ",
                                        input.substr(pos, off - pos)));

    off = input.find_first_not_of(" \t\r\n", off);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no max"));
    pos = off;
    off = input.find_first_of(" \t\r\n", pos);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no max"));
    int64 max;
    OP_REQUIRES(context,
                (strings::safe_strto64(input.substr(pos, off - pos), &max)),
                errors::InvalidArgument("unable to parse max: ",
                                        input.substr(pos, off - pos)));
    OP_REQUIRES(context, (max == 255 || max == 65535),
                errors::InvalidArgument("invalid max value: ", max));

    Tensor* image_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(
                     0, TensorShape({height, width, channels}), &image_tensor));
    if (magic == "P2" || magic == "P3") {
      for (int64 i = 0; i < image_tensor->NumElements(); i++) {
        off = input.find_first_not_of(" \t\r\n", off);
        OP_REQUIRES(context, (off != string::npos),
                    errors::InvalidArgument("not enough value"));
        pos = off;
        off = input.find_first_of(" \t\r\n", pos);
        OP_REQUIRES(context, (off != string::npos),
                    errors::InvalidArgument("no value"));
        int32 value;
        OP_REQUIRES(
            context,
            (strings::safe_strto32(input.substr(pos, off - pos), &value)),
            errors::InvalidArgument("unable to parse value: ",
                                    input.substr(pos, off - pos)));
        if (image_tensor->dtype() == DT_UINT8) {
          if (max == 255) {
            image_tensor->flat<uint8>()(i) = static_cast<uint8>(value);
          } else {
            image_tensor->flat<uint8>()(i) = static_cast<uint8>(value / 256);
          }
        } else {
          if (max == 255) {
            image_tensor->flat<uint16>()(i) = static_cast<uint16>(value * 256);
          } else {
            image_tensor->flat<uint16>()(i) = static_cast<uint16>(value);
          }
        }
      }
    } else {
      off++;
      if (image_tensor->dtype() == DT_UINT8) {
        if (max == 255) {
          OP_REQUIRES(context,
                      (off + image_tensor->NumElements() <= input.size()),
                      errors::InvalidArgument("not enough data"));
          memcpy(image_tensor->flat<uint8>().data(), &input[off],
                 image_tensor->NumElements());
        } else {
          // TODO: add support for max = 65535 and dtype = uint8; need test file
          OP_REQUIRES(
              context, false,
              errors::InvalidArgument(
                  "not supported with max == 65535 and dtype == uint8"));
        }
      } else {
        if (max == 255) {
          // TODO: add support for max = 255 and dtype = uint16; need test file
          OP_REQUIRES(context, false,
                      errors::InvalidArgument(
                          "not supported with max == 255 and dtype == uint16"));
        } else {
          // network order so switch
          for (int64 i = 0; i < image_tensor->NumElements(); i++) {
            image_tensor->flat<uint16>()(i) =
                static_cast<uint16>((((int32)input[off + i * 2] & 0xFF) << 8) |
                                    (((int32)input[off + i * 2 + 1] & 0xFF)));
          }
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("IO>DecodePnm").Device(DEVICE_CPU), DecodePNMOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
