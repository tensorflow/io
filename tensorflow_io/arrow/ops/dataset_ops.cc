/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

REGISTER_OP("ArrowDataset")
    .Input("serialized_batches: string")
    .Input("columns: int32")
    .Input("batch_size: int64")
    .Input("batch_mode: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that reads serialized Arrow RecordBatches in file format.

serialized_batches: Serialized Arrow RecordBatches.
)doc");

REGISTER_OP("ArrowFeatherDataset")
    .Input("filenames: string")
    .Input("columns: int32")
    .Input("batch_size: int64")
    .Input("batch_mode: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that reads files in Arrow Feather format.

filenames: One or more file paths.
)doc");

REGISTER_OP("ArrowStreamDataset")
    .Input("endpoints: string")
    .Input("columns: int32")
    .Input("batch_size: int64")
    .Input("batch_mode: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that connects to a host serving Arrow RecordBatches in stream format.

endpoints: One or more host addresses that are serving an Arrow stream.
)doc");


REGISTER_OP("ListFeatherColumns")
    .Input("filename: string")
    .Input("memory: string")
    .Output("columns: string")
    .Output("dtypes: string")
    .Output("shapes: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
       c->set_output(0, c->MakeShape({c->UnknownDim()}));
       c->set_output(1, c->MakeShape({c->UnknownDim()}));
       c->set_output(2, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
       return Status::OK();
     });

}  // namespace tensorflow
