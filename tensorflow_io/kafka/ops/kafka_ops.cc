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

REGISTER_OP("KafkaIndexableInit")
  .Input("input: string")
  .Input("metadata: string")
  .Input("iterable: resource")
  .Output("resource: resource")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
   });

REGISTER_OP("KafkaIndexableSpec")
  .Input("input: resource")
  .Input("component: int64")
  .Output("shape: int64")
  .Output("dtype: int64")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->MakeShape({c->UnknownDim()}));
    c->set_output(1, c->MakeShape({}));
    return Status::OK();
   });

REGISTER_OP("KafkaIndexableRead")
  .Input("input: resource")
  .Input("start: int64")
  .Input("stop: int64")
  .Input("component: int64")
  .Output("value: dtype")
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

REGISTER_OP("KafkaIterableInit")
  .Input("input: string")
  .Input("metadata: string")
  .Output("resource: resource")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
   });

REGISTER_OP("KafkaIterableNext")
  .Input("input: resource")
  .Input("capacity: int64")
  .Input("component: int64")
  .Output("value: dtype")
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


REGISTER_OP("KafkaOutputSequence")
    .Input("topic: string")
    .Input("servers: string")
    .Output("sequence: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      c->set_output(0, c->Scalar());
      return Status::OK();
    });
REGISTER_OP("KafkaOutputSequenceSetItem")
    .Input("sequence: resource")
    .Input("index: int64")
    .Input("item: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);
REGISTER_OP("KafkaOutputSequenceFlush")
    .Input("sequence: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

}  // namespace tensorflow
