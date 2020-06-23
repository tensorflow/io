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

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace io {
namespace {

struct VectorHasher {
  int operator()(const absl::InlinedVector<int64, 4>& v) const {
    int hash = v.size();
    for (auto& i : v) {
      hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

class OrderIndicesOp : public OpKernel {
 public:
  explicit OrderIndicesOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    OP_REQUIRES(context, input_tensor.dims() == 2,
                errors::InvalidArgument("input must be 2D, received: ",
                                        input_tensor.dims()));

    const Tensor& shape_tensor = context->input(1);
    OP_REQUIRES(context, shape_tensor.dims() == 1,
                errors::InvalidArgument("shape must be 1D, received: ",
                                        shape_tensor.dims()));

    const Tensor& axis_tensor = context->input(2);
    const int64 axis = axis_tensor.scalar<int64>()();
    OP_REQUIRES(context, axis < shape_tensor.NumElements(),
                errors::InvalidArgument("axis must be within [0, ",
                                        shape_tensor.NumElements(), ")"));
    OP_REQUIRES(context, shape_tensor.NumElements() == input_tensor.dim_size(1),
                errors::InvalidArgument("shape size must equal to ",
                                        input_tensor.dim_size(1), ")"));
    std::unordered_map<absl::InlinedVector<int64, 4>,
                       absl::InlinedVector<int64, 4>, VectorHasher>
        entries;
    for (int64 i = 0; i < input_tensor.dim_size(0); i++) {
      absl::InlinedVector<int64, 4> k;
      k.reserve(input_tensor.dim_size(1));
      for (int64 j = 0; j < input_tensor.dim_size(1); j++) {
        k.push_back(input_tensor.matrix<int64>()(i, j));
      }
      k[axis] = 0;

      const auto& lookup = entries.find(k);
      if (lookup == entries.end()) {
        entries[k] = absl::InlinedVector<int64, 4>();
      }
      entries[k].push_back(input_tensor.matrix<int64>()(i, axis));
    }

    int64 max_size = 0;
    for (const auto& lookup : entries) {
      max_size =
          (max_size > lookup.second.size()) ? max_size : lookup.second.size();
    }

    absl::InlinedVector<int64, 4> dims;
    dims.reserve(shape_tensor.NumElements());
    for (int64 i = 0; i < shape_tensor.NumElements(); i++) {
      dims.push_back(shape_tensor.flat<int64>()(i));
    }
    dims[axis] = max_size;
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(dims),
                                                     &output_tensor));
    for (int64 i = 0; i < output_tensor->NumElements(); i++) {
      output_tensor->flat<int64>()(i) = -1;
    }

    absl::InlinedVector<int64, 4> multiplier(dims.size(), 1);
    for (int64 i = multiplier.size() - 2; i >= 0; i--) {
      multiplier[i] = multiplier[i + 1] * dims[i + 1];
    }

    for (const auto& lookup : entries) {
      absl::InlinedVector<int64, 4> index(lookup.first.begin(),
                                          lookup.first.end());
      for (int64 e = 0; e < lookup.second.size(); e++) {
        index[axis] = e;
        int64 offset = 0;
        for (int64 ii = 0; ii < index.size(); ii++) {
          offset += index[ii] * multiplier[ii];
        }
        output_tensor->flat<int64>()(offset) = lookup.second[e];
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("IO>OrderIndices").Device(DEVICE_CPU),
                        OrderIndicesOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
