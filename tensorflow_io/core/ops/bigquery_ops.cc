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

namespace tensorflow {

REGISTER_OP("IO>BigQueryClient")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("client: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IO>BigQueryReadSession")
    .Input("client: resource")
    .Attr("parent: string")
    .Attr("project_id: string")
    .Attr("table_id: string")
    .Attr("dataset_id: string")
    .Attr("selected_fields: list(string) >= 1")
    .Attr("output_types: list(type) >= 1")
    .Attr("default_values: list(string) >= 1")
    .Attr("requested_streams: int")
    .Attr("data_format: string")
    .Attr("row_restriction: string = ''")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("streams: string")
    .Output("schema: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->Scalar());
      return tensorflow::Status::OK();
    });

REGISTER_OP("IO>BigQueryDataset")
    .Input("client: resource")
    .Input("stream: string")
    .Input("schema: string")
    .Attr("offset: int")
    .Attr("data_format: string")
    .Attr("selected_fields: list(string) >= 1")
    .Attr("output_types: list(type) >= 1")
    .Attr("default_values: list(string) >= 1")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

}  // namespace tensorflow
