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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace io {
namespace {

REGISTER_OP("IO>OrderIndices")
    .Input("input: int64")
    .Input("shape: int64")
    .Input("axis: int64")
    .Output("value: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &shape));
      if (c->RankKnown(shape)) {
        std::vector<shape_inference::DimensionHandle> dims;
        dims.reserve(c->Rank(shape));
        for (int i = 0; i < c->Rank(shape); ++i) {
          dims.emplace_back(c->UnknownDim());
        }
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
      }

      c->set_output(0, c->UnknownShape());
      return Status::OK();
    });

}  // namespace
}  // namespace io
}  // namespace tensorflow
