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

REGISTER_OP("IO>NumpyInfo")
    .Input("filename: string")
    .Output("array: string")
    .Output("shape: int64")
    .Output("dtype: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
      c->set_output(2, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>NumpyRead")
    .Input("address: int64")
    .Input("filename: string")
    .Input("array: string")
    .Input("shape: int64")
    .Input("start: int64")
    .Input("stop: int64")
    .Attr("dtype: type")
    .Output("output: dtype")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle full;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &full));
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->ReplaceDim(full, 0, c->UnknownDim(), &shape));
      c->set_output(0, shape);
      return Status::OK();
    });

}  // namespace
}  // namespace io
}  // namespace tensorflow
