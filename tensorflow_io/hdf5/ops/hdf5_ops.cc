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

REGISTER_OP("HDF5IndexableInit")
  .Input("input: string")
  .Output("output: resource")
  .Output("component: string")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    c->set_output(1, c->MakeShape({}));
    return Status::OK();
   });
REGISTER_OP("HDF5IndexableSpec")
  .Input("input: resource")
  .Input("component: string")
  .Output("shape: int64")
  .Output("dtype: int64")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->MakeShape({c->UnknownDim()}));
    c->set_output(1, c->MakeShape({}));
    return Status::OK();
   });

REGISTER_OP("HDF5IndexableGetItem")
  .Input("input: resource")
  .Input("start: int64")
  .Input("stop: int64")
  .Input("step: int64")
  .Input("component: string")
  .Output("output: dtype")
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

REGISTER_OP("ListHDF5Datasets")
    .Input("filename: string")
    .Input("memory: string")
    .Output("datasets: string")
    .Output("dtypes: string")
    .Output("shapes: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
       c->set_output(0, c->MakeShape({c->UnknownDim()}));
       c->set_output(1, c->MakeShape({c->UnknownDim()}));
       c->set_output(2, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
       return Status::OK();
     });

REGISTER_OP("ReadHDF5")
    .Input("filename: string")
    .Input("dataset: string")
    .Input("memory: string")
    .Input("start: int64")
    .Input("stop: int64")
    .Attr("dtype: type")
    .Output("output: dtype")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    });

}  // namespace tensorflow
