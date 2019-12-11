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
#include "tensorflow_io/avro/utils/parse_avro_attrs.h"

namespace tensorflow {

using ::tensorflow::shape_inference::ShapeHandle;

// Adjusted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/dataset_ops.cc
REGISTER_OP("AvroRecordDataset")
    .Input("filenames: string")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      VLOG(4) << "Create avro record dataset";
      // `filenames` must be a scalar or a vector
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      // `buffer_size` must be a scalar
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });


REGISTER_OP("AvroDataset")
    .Input("filenames: string")
    .Input("batch_size: int64")
    .Input("drop_remainder: bool")
    .Input("dense_defaults: dense_types")
    .Input("input_stream_buffer_size: int64")
    .Input("avro_data_buffer_size: int64")
    .Output("handle: variant")
    .Attr("reader_schema: string")
    .Attr("sparse_keys: list(string) >= 0")
    .Attr("dense_keys: list(string) >= 0")
    .Attr("sparse_types: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("dense_types: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
                                              // Output components will be
                                              // sorted by key (dense_keys and
                                              // sparse_keys combined) here.
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 num_dense;
      std::vector<DataType> sparse_types;
      std::vector<DataType> dense_types;
      std::vector<PartialTensorShape> dense_shapes;

      TF_RETURN_IF_ERROR(c->GetAttr("sparse_types", &sparse_types));
      TF_RETURN_IF_ERROR(c->GetAttr("dense_types", &dense_types));
      TF_RETURN_IF_ERROR(c->GetAttr("dense_shapes", &dense_shapes));

      num_dense = dense_types.size();

      // Add input checking
      if (static_cast<size_t>(num_dense) != dense_shapes.size()) {
        return errors::InvalidArgument("len(dense_keys) != len(dense_shapes)");
      }
      if (num_dense > std::numeric_limits<int32>::max()) {
        return errors::InvalidArgument("num_dense_ too large");
      }
      for (const DataType& type : dense_types) {
        TF_RETURN_IF_ERROR(data::CheckValidType(type));
      }
      for (const DataType& type : sparse_types) {
        TF_RETURN_IF_ERROR(data::CheckValidType(type));
      }

      // Log schema if the user provided one at op kernel construction
      string schema;
      TF_RETURN_IF_ERROR(c->GetAttr("reader_schema", &schema));
      if (schema.size()) {
        VLOG(4) << "Avro parser for reader schema\n" << schema;
      } else {
        VLOG(4) << "Avro parser with default schema";
      }

      return shape_inference::ScalarShape(c);
    });

}  // namespace tensorflow
