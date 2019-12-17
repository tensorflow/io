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

REGISTER_OP("IO>EncodeAvro")
  .Input("input: dtype")
  .Output("string: string")
  .Attr("schema: string")
  .Attr("dtype: list({float,double,int32,int64,string})")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
   });

REGISTER_OP("IO>DecodeAvroInit")
  .Input("input: string")
  .Output("resource: resource")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
   });

REGISTER_OP("IO>DecodeAvro")
  .Input("input: string")
  .Input("schema: T")
  .Output("value: dtype")
  .Attr("dtype: list({float,double,int32,int64,string})")
  .Attr("T: {string, resource}")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    for (int64 i = 0; i < c->num_outputs(); i++) {
      c->set_output(i, c->input(0));
    }
    return Status::OK();
   });

REGISTER_OP("IO>KafkaReadableInit")
  .Input("input: string")
  .Input("metadata: string")
  .Output("resource: resource")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
   });

REGISTER_OP("IO>KafkaReadableRead")
  .Input("input: resource")
  .Input("start: int64")
  .Input("stop: int64")
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

REGISTER_OP("IO>KafkaOutputSequence")
    .Input("topic: string")
    .Input("metadata: string")
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

REGISTER_OP("IO>KafkaOutputSequenceSetItem")
    .Input("sequence: resource")
    .Input("index: int64")
    .Input("item: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IO>KafkaOutputSequenceFlush")
    .Input("sequence: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);



REGISTER_OP("IO>LayerKafkaCall")
  .Input("input: T")
  .Input("content: string")
  .Input("resource: resource")
  .Output("output: T")
  .Attr("T: type")
  .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("IO>LayerKafkaInit")
  .Input("topic: string")
  .Input("partition: int32")
  .Input("metadata: string")
  .Output("resource: resource")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IO>LayerKafkaSync")
  .Input("resource: resource")
  .SetShapeFn(shape_inference::ScalarShape);

}  // namespace tensorflow
