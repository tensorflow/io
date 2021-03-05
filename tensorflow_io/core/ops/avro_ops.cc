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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow_io/core/kernels/avro/utils/parse_avro_attrs.h"

namespace tensorflow {

using ::tensorflow::shape_inference::ShapeHandle;

// Copied verbatim from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/parsing_ops.cc
// since this is not exposed publicly
// Adds output shapes for dense tensors in Parse*Example ops.
template <typename TensorShapeType>  // TensorShape or PartialTensorShape
Status AddDenseOutputShapes(const std::vector<TensorShapeType>& dense_shapes,
                            const ShapeHandle& prefix,
                            shape_inference::InferenceContext* c,
                            int* output_idx) {
  for (const auto& dense_shape : dense_shapes) {
    ShapeHandle s;
    TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(dense_shape, &s));
    TF_RETURN_IF_ERROR(c->Concatenate(prefix, s, &s));
    c->set_output((*output_idx)++, s);
  }
  return Status::OK();
}

shape_inference::DimensionOrConstant ComputeSparseRank(
    const ShapeHandle input_shape, int64 rank_delta,
    shape_inference::InferenceContext* c) {
  shape_inference::DimensionOrConstant rank(c->UnknownDim());
  if (c->RankKnown(input_shape)) {
    rank = c->Rank(input_shape) + rank_delta;
  }
  return rank;
}

// Copied verbatim from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/parsing_ops.cc
// since this is not exposed publicly
// Adds output shapes for sparse tensors in Parse*Example ops.
void AddSparseOutputShapes(int num_sparse, const ShapeHandle input_shape,
                           std::vector<int64> sparse_ranks,
                           shape_inference::InferenceContext* c,
                           int* output_idx) {
  for (int i = 0; i < num_sparse; ++i) {  // sparse_indices
    shape_inference::DimensionOrConstant rank =
        ComputeSparseRank(input_shape, sparse_ranks[i], c);
    c->set_output((*output_idx)++, c->Matrix(c->UnknownDim(), rank));
  }
  for (int i = 0; i < num_sparse; ++i) {  // sparse_values
    c->set_output((*output_idx)++, c->Vector(c->UnknownDim()));
  }
  for (int i = 0; i < num_sparse; ++i) {  // sparse_dense_shapes
    shape_inference::DimensionOrConstant rank =
        ComputeSparseRank(input_shape, sparse_ranks[i], c);
    c->set_output((*output_idx)++, c->Vector(rank));
  }
}

// Adjusted from ParseExample and ParseExampleV2 here
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/parsing_ops.cc
REGISTER_OP("IO>ParseAvro")
    .Input("serialized: string")
    .Input("names: string")
    .Input("dense_defaults: dense_types")
    .Output("sparse_indices: num_sparse * int64")
    .Output("sparse_values: sparse_types")
    .Output("sparse_shapes: num_sparse * int64")
    .Output("dense_values: dense_types")
    .Attr("avro_num_minibatches: int >= 0")
    .Attr("num_sparse: int >= 0")
    .Attr("reader_schema: string")
    .Attr("sparse_keys: list(string) >= 0")
    .Attr("sparse_ranks: list(int) >= 0")
    .Attr("dense_keys: list(string) >= 0")
    .Attr("sparse_types: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("dense_types: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      size_t num_dense;
      size_t num_sparse;
      int64 avro_num_minibatches;
      int64 num_sparse_from_user;
      std::vector<DataType> sparse_types;
      std::vector<DataType> dense_types;
      std::vector<string> sparse_keys;
      std::vector<int64> sparse_ranks;
      std::vector<string> dense_keys;

      std::vector<PartialTensorShape> dense_shapes;

      TF_RETURN_IF_ERROR(c->GetAttr("sparse_types", &sparse_types));
      TF_RETURN_IF_ERROR(c->GetAttr("dense_types", &dense_types));
      TF_RETURN_IF_ERROR(c->GetAttr("dense_shapes", &dense_shapes));
      TF_RETURN_IF_ERROR(
          c->GetAttr("avro_num_minibatches", &avro_num_minibatches));

      TF_RETURN_IF_ERROR(c->GetAttr("sparse_keys", &sparse_keys));
      TF_RETURN_IF_ERROR(c->GetAttr("sparse_ranks", &sparse_ranks));
      TF_RETURN_IF_ERROR(c->GetAttr("dense_keys", &dense_keys));

      TF_RETURN_IF_ERROR(c->GetAttr("num_sparse", &num_sparse_from_user));

      num_sparse = sparse_types.size();
      num_dense = dense_types.size();

      // Check input parameters
      if (num_sparse != static_cast<size_t>(num_sparse_from_user)) {
        return errors::InvalidArgument("len(sparse_types) != num_sparse");
      }
      if (num_sparse != sparse_keys.size()) {
        return errors::InvalidArgument("len(sparse_types) != len(sparse_keys)");
      }
      if (num_sparse != sparse_ranks.size()) {
        return errors::InvalidArgument("len(sparse_ranks) != num_sparse");
      }
      if (num_dense != dense_keys.size()) {
        return errors::InvalidArgument("len(dense_types) != len(dense_keys)");
      }
      if (num_dense != dense_shapes.size()) {
        return errors::InvalidArgument("len(dense_types) != len(dense_shapes)");
      }
      if (num_dense > std::numeric_limits<int32>::max()) {
        return errors::InvalidArgument("num_dense too large");
      }
      for (const DataType& type : dense_types) {
        TF_RETURN_IF_ERROR(data::CheckValidType(type));
      }
      for (const DataType& type : sparse_types) {
        TF_RETURN_IF_ERROR(data::CheckValidType(type));
      }

      // Log schema if the user provided one at op kernel construction
      string reader_schema_str;
      TF_RETURN_IF_ERROR(c->GetAttr("reader_schema", &reader_schema_str));
      if (reader_schema_str.empty()) {
        return errors::InvalidArgument(
            "User must provide a valid reader_schema_str, got ",
            reader_schema_str);
      }

      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      ShapeHandle names;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &names));

      int output_idx = 0;
      AddSparseOutputShapes(num_sparse, input, sparse_ranks, c, &output_idx);
      TF_RETURN_IF_ERROR(
          AddDenseOutputShapes(dense_shapes, input, c, &output_idx));
      return Status::OK();
    });

// Adjusted from TFRecordDataset here
// https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/core/ops/dataset_ops.cc
REGISTER_OP("IO>AvroRecordDataset")
    .Input("filenames: string")
    .Input("buffer_size: int64")
    .Input("reader_schema: string")
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
      // `reader_schema` must be a scalar
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("IO>AvroDataset")
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

REGISTER_OP("IO>ListAvroColumns")
    .Input("filename: string")
    .Input("schema: string")
    .Input("memory: string")
    .Output("columns: string")
    .Output("dtypes: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>ReadAvro")
    .Input("filename: string")
    .Input("schema: string")
    .Input("column: string")
    .Input("memory: string")
    .Input("offset: int64")
    .Input("length: int64")
    .Attr("dtype: type")
    .Output("output: dtype")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>AvroReadableInit")
    .Input("input: string")
    .Input("metadata: string")
    .Output("resource: resource")
    .Output("components: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->MakeShape({}));
      return Status::OK();
    });

REGISTER_OP("IO>AvroReadableSpec")
    .Input("input: resource")
    .Output("shape: int64")
    .Output("dtype: int64")
    .Attr("component: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->MakeShape({}));
      return Status::OK();
    });

REGISTER_OP("IO>AvroReadableRead")
    .Input("input: resource")
    .Input("start: int64")
    .Input("stop: int64")
    .Output("value: dtype")
    .Attr("component: string")
    .Attr("filter: list(string) = []")
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

REGISTER_OP("IO>AvroReadablePartitions")
    .Input("input: resource")
    .Output("partitions: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

}  // namespace tensorflow
