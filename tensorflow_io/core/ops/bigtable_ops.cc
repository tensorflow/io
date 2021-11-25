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
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a BigtableClientResource representing a connection to Google Bigtable.
)doc");

REGISTER_OP("BigtableDataset")
    .Input("client: resource")
    .Input("row_set: resource")
    .Input("filter: resource")
    .Attr("table_id: string")
    .Attr("columns: list(string) >= 1")
    .Attr("output_type: type")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a BigtableDataset used for iterating through values from the specified
table.

row_set: BigtableRowSetResource representing a RowSet the user wishes to 
retrieve.
table_id: ID of the table user wants to read from.
columns: List of names of the columns user wants to retrieve in format 
'column_family:column_name'
)doc");

REGISTER_OP("BigtableEmptyRowSet")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("row_set: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a BigtableRowSetResource representing empty RowSet.
)doc");

REGISTER_OP("BigtableEmptyRowRange")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("row_range: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a BigtableRowRangeResource representing empty RowRange.
)doc");

REGISTER_OP("BigtablePrefixRowRange")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("prefix: string")
    .Output("row_range: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a BigtableRowRangeResource representing RowRange of all RowKeys 
starting with the given prefix.
)doc");

REGISTER_OP("BigtableRowRange")
    .Attr("left_row_key: string")
    .Attr("left_open: bool")
    .Attr("right_row_key: string")
    .Attr("right_open: bool")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("row_range: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a BigtableRowRangeResource representing RowRange starting at 
`left_row_key`, ending at `right_row_key`. If left or right row_key is empty, 
the range is assumed to continue infinitely.

left_open: specifies whether to exclude `left_row_key`.
right_open: specifies whether to exclude `right_row_key`.
)doc");

REGISTER_OP("BigtablePrintRowRange")
    .Input("row_range: resource")
    .Output("row_range_str: string")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Returns a string representing a RowRange.
)doc");

REGISTER_OP("BigtablePrintRowSet")
    .Input("row_set: resource")
    .Output("row_set_str: string")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Returns a string representing a RowSet.
)doc");

REGISTER_OP("BigtableRowSetAppendRow")
    .Attr("row_key: string")
    .Input("row_set: resource")
    .Doc(R"doc(
Modifies the RowSet to include one more row.

row_set: BigtableRowSetResource to append to
row_key: string representing a RowKey to append
)doc");

REGISTER_OP("BigtableRowSetAppendRowRange")
    .Input("row_set: resource")
    .Input("row_range: resource")
    .Doc(R"doc(
Modifies the RowSet to include one more range.

row_set: BigtableRowSetResource to append to
row_range: BigtableRowRangeResource to append
)doc");

REGISTER_OP("BigtableRowSetIntersect")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Input("row_set: resource")
    .Input("row_range: resource")
    .Output("result_row_set: resource")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Intersects the RowSet with a RowRange and returns a new RowSet.

row_set: BigtableRowSetResource to intersect
row_range: BigtableRowRangeResource to intersect with
result_row_set: result of the intersection
)doc");

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
    })
    .Doc(R"doc(
Retrieves SampleRowKeys from bigtable, checks which tablets contain row keys 
from the row_set specified by the user and returns a RowSet that represents 
chunks of work for each worker.

client: BigtableClientResource.
row_set: BigtableRowSetResource representing the RowSet specified by the user.
table_id: ID of the table user intends to read from.
num_splits: number of workers between who we split the work.
samples: Tensor of RowSets representing chunks of work.
)doc");

REGISTER_OP("BigtableLatestFilter")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("filter: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a BigtableFilterResource representing a Filter passing only the latest 
value.
)doc");

REGISTER_OP("BigtableTimestampRangeFilter")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("start_ts_us: int")
    .Attr("end_ts_us: int")
    .Output("filter: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a BigtableFilterResource representing Filter passing values created 
between `start_ts_us` and `end_ts_us`.

start_ts_us: The start of the row range (inclusive) in microseconds since epoch.
end_ts_us: The end of the row range (exclusive) in microseconds since epoch.
)doc");

REGISTER_OP("BigtablePrintFilter")
    .Input("filter: resource")
    .Output("output: string")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Returns a string representing the filter.
)doc");
