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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("DecodeTiffInfo")
  .Input("input: string")
  .Output("shape: int64")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle unused;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
    c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
    return Status::OK();
  });

REGISTER_OP("DecodeTiff")
  .Input("input: string")
  .Input("index: int64")
  .Output("image: uint8")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle unused;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
    c->set_output(0, c->MakeShape({
        c->UnknownDim(), c->UnknownDim(), c->UnknownDim()}));
    return Status::OK();
  });

}  // namespace tensorflow
