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
#include "re2/re2.h"

namespace tensorflow {

REGISTER_OP("IO>RE2FullMatch")
    .Input("input: string")
    .Output("output: bool")
    .Output("groups: string")
    .Attr("pattern: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      if (!c->RankKnown(c->input(0))) {
        c->set_output(0, c->UnknownShape());
        c->set_output(1, c->UnknownShape());
        return Status::OK();
      }
      string pattern;
      TF_RETURN_IF_ERROR(c->GetAttr("pattern", &pattern));
      RE2 re(pattern, RE2::Quiet);
      if (!re.ok()) {
        return errors::InvalidArgument("unable to compile pattern '", pattern, "': ", re.error());
      }
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 0, &shape));
      shape_inference::ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(c->Concatenate(shape, c->Vector(re.NumberOfCapturingGroups()), &output_shape));
      c->set_output(0, c->input(0));
      c->set_output(1, output_shape);
      return Status::OK();
    });

REGISTER_OP("IO>LayerTextCall")
  .Input("input: T")
  .Input("content: string")
  .Input("resource: resource")
  .Output("output: T")
  .Attr("T: type")
  .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("IO>LayerTextInit")
  .Input("input: string")
  .Output("resource: resource")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IO>LayerTextSync")
  .Input("resource: resource")
  .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IO>ReadText")
    .Input("filename: string")
    .Input("memory: string")
    .Input("offset: int64")
    .Input("length: int64")
    .Output("output: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
       c->set_output(0, c->MakeShape({c->UnknownDim()}));
       return Status::OK();
     });

REGISTER_OP("IO>TextOutputSequence")
    .Input("destination: string")
    .Output("sequence: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IO>TextOutputSequenceSetItem")
    .Input("sequence: resource")
    .Input("index: int64")
    .Input("item: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IO>CSVReadableInit")
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

REGISTER_OP("IO>CSVReadableSpec")
  .Input("input: resource")
  .Output("shape: int64")
  .Output("dtype: int64")
  .Attr("component: string")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->MakeShape({c->UnknownDim()}));
    c->set_output(1, c->MakeShape({}));
    return Status::OK();
   });

REGISTER_OP("IO>CSVReadableRead")
  .Input("input: resource")
  .Input("start: int64")
  .Input("stop: int64")
  .Output("value: dtype")
  .Attr("filter: list(string) = []")
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
