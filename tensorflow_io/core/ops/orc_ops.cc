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
REGISTER_OP("IO>ORCReadableInit")
    .Input("input: string")
    .Output("resource: resource")
    .Output("components: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->MakeShape({}));
      return Status::OK();
    });

REGISTER_OP("IO>ORCReadableSpec")
    .Input("input: resource")
    .Output("shape: int64")
    .Output("dtype: int64")
    .Attr("component: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->MakeShape({}));
      return Status::OK();
    });

REGISTER_OP("IO>ORCReadableRead")
    .Input("input: resource")
    .Input("start: int64")
    .Input("stop: int64")
    .Output("value: dtype")
    .Attr("component: string")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));
      shape_inference::ShapeHandle entry;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape, &entry));
      c->set_output(0, entry);
      return Status::OK();
    });
}  // namespace tensorflow