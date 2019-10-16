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

// TODO(saeta): Add support for setting ClientOptions values.
REGISTER_OP("IoBigtableClient")
    .Attr("project_id: string")
    .Attr("instance_id: string")
    .Attr("connection_pool_size: int")
    .Attr("max_receive_message_size: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("client: resource")
    .SetShapeFn(shape_inference::ScalarShape);

// TODO(saeta): Add support for Application Profiles.
// See https://cloud.google.com/bigtable/docs/app-profiles for more info.
REGISTER_OP("IoBigtableTable")
    .Input("client: resource")
    .Attr("table_name: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("table: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IoDatasetToBigtable")
    .Input("table: resource")
    .Input("input_dataset: variant")
    .Input("column_families: string")
    .Input("columns: string")
    .Input("timestamp: int64")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("IoBigtableLookupDataset")
    .Input("keys_dataset: variant")
    .Input("table: resource")
    .Input("column_families: string")
    .Input("columns: string")
    .Output("handle: variant")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IoBigtablePrefixKeyDataset")
    .Input("table: resource")
    .Input("prefix: string")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IoBigtableRangeKeyDataset")
    .Input("table: resource")
    .Input("start_key: string")
    .Input("end_key: string")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IoBigtableSampleKeysDataset")
    .Input("table: resource")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IoBigtableSampleKeyPairsDataset")
    .Input("table: resource")
    .Input("prefix: string")
    .Input("start_key: string")
    .Input("end_key: string")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

// TODO(saeta): Support continuing despite bad data (e.g. empty string, or
// skip incomplete row.)
REGISTER_OP("IoBigtableScanDataset")
    .Input("table: resource")
    .Input("prefix: string")
    .Input("start_key: string")
    .Input("end_key: string")
    .Input("column_families: string")
    .Input("columns: string")
    .Input("probability: float")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

}  // namespace tensorflow
