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

REGISTER_OP("InitLMDB")
  .Input("input: string")
  .Input("memory: string")
  .Input("metadata: string")
  .Output("output: resource")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .SetIsStateful()
  .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("NextLMDB")
  .Input("input: resource")
  .Input("capacity: int64")
  .Output("output: dtypes")
  .Attr("dtypes: list(type) >= 1")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    for (int64 i = 0; i < c->num_outputs(); i++) {
      c->set_output(i, c->MakeShape({c->UnknownDim()}));
    }
    return Status::OK();
   });

REGISTER_OP("ReadLMDB")
  .Input("input: string")
  .Input("memory: string")
  .Input("metadata: string")
  .Output("output: dtypes")
  .Attr("dtypes: list(type) >= 1")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    for (int64 i = 0; i < c->num_outputs(); i++) {
      c->set_output(i, c->MakeShape({c->UnknownDim()}));
    }
    return Status::OK();
   });
}  // namespace tensorflow
