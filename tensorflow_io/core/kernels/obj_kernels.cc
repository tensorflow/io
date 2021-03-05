/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/platform/logging.h"
#include "tiny_obj_loader.h"

namespace tensorflow {
namespace io {
namespace {

class DecodeObjOp : public OpKernel {
 public:
  explicit DecodeObjOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(input_tensor->shape()),
                errors::InvalidArgument("input must be scalar, got shape ",
                                        input_tensor->shape().DebugString()));
    const tstring& input = input_tensor->scalar<tstring>()();

    tinyobj::ObjReader reader;

    if (!reader.ParseFromString(input.c_str(), "")) {
      OP_REQUIRES(
          context, false,
          errors::Internal("Unable to read obj file: ", reader.Error()));
    }

    if (!reader.Warning().empty()) {
      LOG(WARNING) << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();

    int64 count = attrib.vertices.size() / 3;

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({count, 3}),
                                                     &output_tensor));
    // Loop over attrib.vertices:
    for (int64 i = 0; i < count; i++) {
      tinyobj::real_t x = attrib.vertices[i * 3 + 0];
      tinyobj::real_t y = attrib.vertices[i * 3 + 1];
      tinyobj::real_t z = attrib.vertices[i * 3 + 2];
      output_tensor->tensor<float, 2>()(i, 0) = x;
      output_tensor->tensor<float, 2>()(i, 1) = y;
      output_tensor->tensor<float, 2>()(i, 2) = z;
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("IO>DecodeObj").Device(DEVICE_CPU), DecodeObjOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
