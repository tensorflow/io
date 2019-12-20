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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace io {
namespace {

REGISTER_OP("IO>FileInfo")
    .Input("input: string")
    .Output("length: int64")
    .Output("compression: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>FileRead")
    .Input("input: string")
    .Input("offset: int64")
    .Input("length: int64")
    .Input("compression: string")
    .Output("value: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IO>FileInit")
    .SetIsStateful()
    .Input("input: string")
    .Output("resource: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IO>FileCall")
    .SetIsStateful()
    .Input("input: string")
    .Input("final: bool")
    .Input("resource: resource")
    .Output("output: string")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("IO>FileSync")
    .Input("resource: resource")
    .SetShapeFn(shape_inference::ScalarShape);

}  // namespace
}  // namespace io
}  // namespace tensorflow
