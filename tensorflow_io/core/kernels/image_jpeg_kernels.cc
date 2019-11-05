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
#include "exif.h"

namespace tensorflow {
namespace data {
namespace {

class DecodeJpegExifOp : public OpKernel {
 public:
  explicit DecodeJpegExifOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    Tensor* orientation_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &orientation_tensor));
    orientation_tensor->scalar<int64>()() = 0;

    const string& input = input_tensor->scalar<string>()();
    easyexif::EXIFInfo result;
    if (result.parseFrom(input) == PARSE_EXIF_SUCCESS) {
      orientation_tensor->scalar<int64>()() = result.Orientation;
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("IO>DecodeJpegExif").Device(DEVICE_CPU),
                        DecodeJpegExifOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
