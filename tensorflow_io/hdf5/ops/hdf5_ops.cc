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
    .Input("start: int64")
    .Input("count: int64")
    .Input("memory: string")
    .Attr("dtype: type")
    .Output("output: dtype")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      if (c->RankKnown(c->input(3))) {
        c->set_output(0, c->UnknownShapeOfRank(c->Rank(c->input(3))));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return Status::OK();
    });

REGISTER_OP("HDF5Input")
    .Input("source: string")
    .Output("handle: variant")
    .Attr("filters: list(string) = []")
    .Attr("columns: list(string) = []")
    .Attr("schema: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
       c->set_output(0, c->MakeShape({c->UnknownDim()}));
       return Status::OK();
     });

REGISTER_OP("HDF5Dataset")
    .Input("input: T")
    .Input("batch: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("T: {string, variant} = DT_VARIANT")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
       c->set_output(0, c->MakeShape({}));
       return Status::OK();
     });

}  // namespace tensorflow
