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

REGISTER_OP("IO>SqlIterableInit")
    .SetIsStateful()
    .Input("input: string")
    .Input("endpoint: string")
    .Output("resource: resource")
    .Output("count: int64")
    .Output("field: string")
    .Output("dtype: int64")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      c->set_output(2, c->MakeShape({}));
      c->set_output(3, c->MakeShape({}));
      return OkStatus();
    });

REGISTER_OP("IO>SqlIterableRead")
    .SetIsStateful()
    .Input("input: resource")
    .Input("index: int64")
    .Input("field: string")
    .Output("value: dtypes")
    .Attr("dtypes: list(type)")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int64 i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, c->MakeShape({c->UnknownDim()}));
      }
      return OkStatus();
    });

}  // namespace
}  // namespace io
}  // namespace tensorflow
