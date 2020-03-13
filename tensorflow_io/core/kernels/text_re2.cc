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
#include "re2/re2.h"

namespace tensorflow {
namespace data {
namespace {

class RE2FullMatchOp : public OpKernel {
 public:
  explicit RE2FullMatchOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pattern", &pattern_));
  }

  void Compute(OpKernelContext* context) override {
    RE2 re(pattern_, RE2::Quiet);
    OP_REQUIRES(context, re.ok(),
                errors::InvalidArgument("unable to compile pattern '", pattern_, "': ", re.error()));

    const Tensor& input_tensor = context->input(0);
      TensorShape shape = input_tensor.shape();

      Tensor* output_tensor;
      OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));

      shape.AddDim(re.NumberOfCapturingGroups());
      Tensor* groups_tensor;
      OP_REQUIRES_OK(context, context->allocate_output(1, shape, &groups_tensor));

      for (int64 i = 0; i < input_tensor.NumElements(); i++) {
        std::vector<RE2::Arg> args(re.NumberOfCapturingGroups());
        std::vector<RE2::Arg *> argv(re.NumberOfCapturingGroups());
        std::vector<string> results(re.NumberOfCapturingGroups());
        for (int j = 0; j < re.NumberOfCapturingGroups(); j++) {
          args[j] = &results[j];
          argv[j] = &args[j];
        }
        string input_string = input_tensor.flat<tstring>()(i);
        output_tensor->flat<bool>()(i) = RE2::FullMatchN(input_string, re, argv.data(), re.NumberOfCapturingGroups());
        if (output_tensor->flat<bool>()(i)) {
          for (int j = 0; j < re.NumberOfCapturingGroups(); j++) {
            groups_tensor->flat<tstring>()(i * re.NumberOfCapturingGroups() + j) = std::move(results[j]);
          }
        }
    }
  }
 private:
  string pattern_;
};

REGISTER_KERNEL_BUILDER(Name("IO>RE2FullMatch").Device(DEVICE_CPU),
                        RE2FullMatchOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
