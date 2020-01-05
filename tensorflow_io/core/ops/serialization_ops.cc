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

REGISTER_OP("IO>DecodeJSON")
    .Input("input: string")
    .Input("names: string")
    .Output("value: dtypes")
    .Attr("shapes: list(shape)")
    .Attr("dtypes: list(type)")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // TODO: support batch (1-D) input
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &unused));
      std::vector<TensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      if (shapes.size() != c->num_outputs()) {
        return errors::InvalidArgument(
            "shapes and types should be the same: ", shapes.size(), " vs. ",
            c->num_outputs());
      }
      for (size_t i = 0; i < shapes.size(); ++i) {
        shape_inference::ShapeHandle shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(shapes[i], &shape));
        c->set_output(static_cast<int64>(i), shape);
      }
      return Status::OK();
    });

REGISTER_OP("IO>DecodeAvroV")
    .Input("input: string")
    .Input("names: string")
    .Input("schema: string")
    .Output("value: dtypes")
    .Attr("shapes: list(shape)")
    .Attr("dtypes: list(type)")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // TODO: support batch (1-D) input
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &unused));
      std::vector<TensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      if (shapes.size() != c->num_outputs()) {
        return errors::InvalidArgument(
            "shapes and types should be the same: ", shapes.size(), " vs. ",
            c->num_outputs());
      }
      for (size_t i = 0; i < shapes.size(); ++i) {
        shape_inference::ShapeHandle shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(shapes[i], &shape));
        c->set_output(static_cast<int64>(i), shape);
      }
      return Status::OK();
    });

REGISTER_OP("IO>EncodeAvroV")
    .Input("input: dtype")
    .Input("names: string")
    .Input("schema: string")
    .Output("value: string")
    .Attr("dtype: list({bool,int32,int64,float,double,string})")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace
}  // namespace io
}  // namespace tensorflow
