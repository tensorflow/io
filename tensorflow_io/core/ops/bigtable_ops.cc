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
    .Attr("table_id: string")
    .Attr("columns: list(string) >= 1")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);