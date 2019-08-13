/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

REGISTER_OP("InitMNIST")
  .Input("input: string")
  .Input("memory: string")
  .Input("metadata: string")
  .Output("output: resource")
  .Output("dtype: int64")
  .Output("shape: int64")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    c->set_output(1, c->Scalar());
    c->set_output(2, c->MakeShape({c->UnknownDim()}));
    return Status::OK();
   });

REGISTER_OP("GetItemMNIST")
  .Input("input: resource")
  .Input("start: int64")
  .Input("stop: int64")
  .Input("step: int64")
  .Output("output: uint8")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
   });

REGISTER_OP("LenMNIST")
  .Input("input: resource")
  .Output("output: int64")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
   });



REGISTER_OP("ReadMNISTLabel")
  .Input("input: string")
  .Input("memory: string")
  .Input("metadata: string")
  .Input("start: int64")
  .Input("stop: int64")
  .Output("output: dtypes")
  .Attr("dtypes: list(type) >= 1")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->MakeShape({c->UnknownDim()}));
    return Status::OK();
   });

REGISTER_OP("ReadMNISTImage")
  .Input("input: string")
  .Input("memory: string")
  .Input("metadata: string")
  .Input("start: int64")
  .Input("stop: int64")
  .Output("output: dtypes")
  .Attr("dtypes: list(type) >= 1")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim(), c->UnknownDim()}));
    return Status::OK();
   });

}  // namespace tensorflow
