/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

using namespace tensorflow;

REGISTER_OP("BigtableClient")
    .Attr("project_id: string")
    .Attr("instance_id: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("client: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BigtableDataset")
    .Input("client: resource")
    .Input("row_set: resource")
    .Input("filter: resource")
    .Attr("table_id: string")
    .Attr("columns: list(string) >= 1")
    .Attr("output_type: type")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BigtableEmptyRowSet")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("row_set: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BigtableEmptyRowRange")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("row_range: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BigtablePrefixRowRange")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("prefix: string")
    .Output("row_range: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BigtableRowRange")
    .Attr("left_row_key: string")
    .Attr("left_open: bool")
    .Attr("right_row_key: string")
    .Attr("right_open: bool")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("row_range: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BigtablePrintRowRange")
    .Input("row_range: resource")
    .Output("row_range_str: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BigtablePrintRowSet")
    .Input("row_set: resource")
    .Output("row_set_str: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BigtableRowSetAppendRow")
    .Attr("row_key: string")
    .Input("row_set: resource");

REGISTER_OP("BigtableRowSetAppendRowRange")
    .Input("row_set: resource")
    .Input("row_range: resource");

REGISTER_OP("BigtableRowSetIntersect")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Input("row_set: resource")
    .Input("row_range: resource")
    .Output("result_row_set: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BigtableSplitRowSetEvenly")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Input("client: resource")
    .Input("row_set: resource")
    .Attr("table_id: string")
    .Attr("num_splits: int")
    .Output("samples: resource")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      return tensorflow::Status::OK();
    });

REGISTER_OP("BigtableLatestFilter")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("filter: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BigtableTimestampRangeFilter")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("start_ts_us: int")
    .Attr("end_ts_us: int")
    .Output("filter: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BigtablePrintFilter")
    .Input("filter: resource")
    .Output("output: string")
    .SetShapeFn(shape_inference::ScalarShape);
