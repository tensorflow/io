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
#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_io/avro/utils/parse_avro_attrs.h"

namespace tensorflow {
namespace data {

Status CheckDenseShapeToBeDefined(
    const std::vector<PartialTensorShape>& dense_shapes) {
  for (int i = 0; i < dense_shapes.size(); ++i) {
    const PartialTensorShape& dense_shape = dense_shapes[i];
    bool shape_ok = true;
    if (dense_shape.dims() == -1) {
      shape_ok = false;
    } else {
      for (int d = 1; d < dense_shape.dims() && shape_ok; ++d) {
        if (dense_shape.dim_size(d) == -1) {
          shape_ok = false;
        }
      }
    }
    if (!shape_ok) {
      return errors::InvalidArgument(
          "dense_shapes[", i,
          "] has unknown rank or unknown inner dimensions: ",
          dense_shape.DebugString());
    }
  }
  return Status::OK();
}

// As boiler plate I used tensorflow/core/util/example_proto_helper.cc and
// therein "ParseSingleExampleAttrs" and
Status CheckValidType(const DataType& dtype) {
  switch (dtype) {
    case DT_BOOL:
    case DT_INT32:
    case DT_INT64:
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_STRING:
      return Status::OK();
    default:
      return errors::InvalidArgument("Received input dtype: ",
                                     DataTypeString(dtype));
  }
}

// Finishes the initialization for the attributes, which essentially checks that
// the attributes have the correct values.
//
// returns OK if all attributes are valid; otherwise false.
Status ParseAvroAttrs::FinishInit() {
  if (static_cast<size_t>(num_sparse) != sparse_types.size()) {
    return errors::InvalidArgument("len(sparse_keys) != len(sparse_types)");
  }
  if (static_cast<size_t>(num_dense) != dense_infos.size()) {
    return errors::InvalidArgument("len(dense_keys) != len(dense_infos)");
  }
  if (num_dense > std::numeric_limits<int32>::max()) {
    return errors::InvalidArgument("num_dense_ too large");
  }
  for (const DenseInformation& dense_info : dense_infos) {
    TF_RETURN_IF_ERROR(CheckValidType(dense_info.type));
  }
  for (const DataType& type : sparse_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow