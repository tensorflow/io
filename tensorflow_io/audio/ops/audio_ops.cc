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

REGISTER_OP("ListWAVInfo")
    .Input("filename: string")
    .Input("memory: string")
    .Output("dtype: string")
    .Output("shape: int64")
    .Output("rate: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
       c->set_output(0, c->MakeShape({}));
       c->set_output(1, c->MakeShape({2}));
       c->set_output(2, c->MakeShape({}));
       return Status::OK();
     });

REGISTER_OP("ReadWAV")
    .Input("filename: string")
    .Input("memory: string")
    .Input("start: int64")
    .Input("stop: int64")
    .Attr("dtype: type")
    .Output("output: dtype")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
       c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
       return Status::OK();
     });

}  // namespace tensorflow
