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
#include <avro.h> // TODO(fraudies): Remove me when old ops are removed

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow_io/avro/utils/parse_avro_attrs.h"

namespace tensorflow {

using ::tensorflow::shape_inference::ShapeHandle;


REGISTER_OP("AvroInput")
    .Input("source: string")
    .Output("handle: variant")
    .Attr("filters: list(string) = []")
    .Attr("columns: list(string) = []")
    .Attr("schema: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
       c->set_output(0, c->MakeShape({c->UnknownDim()}));
       return Status::OK();
     });

REGISTER_OP("AvroDataset2")
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

REGISTER_OP("AvroDataset")
    .Input("filenames: string")
    .Input("batch_size: int64")
    .Input("drop_remainder: bool")
    .Input("dense_defaults: Tdense")
    .Output("handle: variant")
    .Attr("reader_schema: string")
    .Attr("sparse_keys: list(string) >= 0")
    .Attr("dense_keys: list(string) >= 0")
    .Attr("sparse_types: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("Tdense: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")  // Output components will be
                                              // sorted by key (dense_keys and
                                              // sparse_keys combined) here.
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 num_dense;
      std::vector<DataType> sparse_types;
      std::vector<DataType> dense_types;
      std::vector<PartialTensorShape> dense_shapes;

      TF_RETURN_IF_ERROR(c->GetAttr("sparse_types", &sparse_types));
      TF_RETURN_IF_ERROR(c->GetAttr("Tdense", &dense_types));
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

// Register the avro record dataset operator
REGISTER_OP("AvroRecordDataset")
    .Input("filenames: string")
    .Input("schema: string")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that emits the avro records from one or more files.
filenames: A scalar or vector containing the name(s) of the file(s) to be
  read.
schema: A string used that is used for schema resolution.
)doc");

// Register the parse function when building the shared library
// For the op I used as boiler plate: tensorflow/core/ops/parsing_ops.cc and
// there 'ParseExample'
// For the op kernel I used as boiler plate:
// tensorflow/core/kernels/example_parsing_ops.cc and there 'ExampleParserOp'
// For the compute method I used as boiler plate:
// tensorflow/core/util/example_proto_fast_parsing.cc and there the
//   method 'FastParseExample'

REGISTER_OP("ParseAvroRecord")
    .Input("serialized: string")
    .Input("sparse_keys: Nsparse * string")
    .Input("dense_keys: Ndense * string")
    .Input("dense_defaults: Tdense")
    .Output("sparse_indices: Nsparse * int64")
    .Output("sparse_values: sparse_types")
    .Output("sparse_shapes: Nsparse * int64")
    .Output("dense_values: Tdense")
    .Attr("Nsparse: int >= 0")  // Inferred from sparse_keys
    .Attr("Ndense: int >= 0")   // Inferred from dense_keys
    .Attr("sparse_types: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("Tdense: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .Attr("schema: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

       data::ParseAvroAttrs attrs;
       TF_RETURN_IF_ERROR(attrs.Init(c));

       // Get the batch size and load it into input
       ShapeHandle input;
       TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));

       // Get the schema, parse it, and log it
       string schema;
       TF_RETURN_IF_ERROR(c->GetAttr("schema", &schema));

       std::unique_ptr<avro_schema_t, std::function<void(avro_schema_t*)>>
           p_reader_schema(new avro_schema_t, [](avro_schema_t* ptr) {
             avro_schema_decref(*ptr);
           });
       if (avro_schema_from_json_length(schema.c_str(), schema.length(),
                                        &(*p_reader_schema)) != 0) {
         return errors::InvalidArgument("The provided json schema is invalid. ",
                                        avro_strerror());
       }
       LOG(INFO) << "Avro parser with schema\n" << schema;

       int output_idx = 0;

       // Output sparse_indices, sparse_values, sparse_shapes
       for (int i_sparse = 0; i_sparse < attrs.num_sparse; ++i_sparse) {
         c->set_output(output_idx++, c->Matrix(c->UnknownDim(), 2));
       }
       for (int i_sparse = 0; i_sparse < attrs.num_sparse; ++i_sparse) {
         c->set_output(output_idx++, c->Vector(c->UnknownDim()));
       }
       for (int i_sparse = 0; i_sparse < attrs.num_sparse; ++i_sparse) {
         c->set_output(output_idx++, c->Vector(2));
       }

       // Output dense_values
       for (int i_dense = 0; i_dense < attrs.num_dense; ++i_dense) {
         ShapeHandle dense;
         TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
             attrs.dense_infos[i_dense].shape, &dense));
         TF_RETURN_IF_ERROR(c->Concatenate(input, dense, &dense));
         c->set_output(output_idx++, dense);
       }

       return Status::OK();
     })
    .Doc(R"doc(
      Parses a serialized avro record that follows the supplied schema into typed tensors.
      serialized: A vector containing a batch of binary serialized avro records.
      dense_keys: A list of Ndense string Tensors.
        The keys expected are associated with dense values.
      dense_defaults: A list of Ndense Tensors (some may be empty).
        These defaults can either be fully defined for all values in the tensor or they
        can be defined as padding element that is used whenever the original input data
        misses values to fill out the full tensor.
      dense_shapes: A list of Ndense shapes; the shapes of the dense tensors.
        The number of elements corresponding to dense_key[j]
        must always equal dense_shapes[j].NumEntries().
        If dense_shapes[j] == (D0, D1, ..., DN) then the shape of output
        Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
        The dense outputs are just the inputs row-stacked by batch.
        This works for dense_shapes[j] = (-1, D1, ..., DN).  In this case
        the shape of the output Tensor dense_values[j] will be
        (|serialized|, M, D1, .., DN), where M is the maximum number of blocks
        of elements of length D1 * .... * DN, across all minibatch entries
        in the input.  Any minibatch entry with less than M blocks of elements of
        length D1 * ... * DN will be padded with the corresponding default_value
        scalar element along the second dimension.
      dense_types: A list of Ndense types; the type of data in each Feature given in dense_keys.
      sparse_keys: A list of Ndense string Tensors (scalars).
        The keys expected are associated with sparse values.
      sparse_types: A list of Nsparse types; the data types of data in each Feature
        given in sparse_keys.
      schema: A string that describes the avro schema of the underlying serialized avro string.
        Currently the parse function supports the primitive types DT_STRING, DT_DOUBLE, DT_FLOAT,
        DT_INT64, DT_INT32, and DT_BOOL.
        The supported avro version depends on the compiled library avro library linked against
        TensorFlow during build. More instructions on avro:
        https://avro.apache.org/docs/1.8.1/spec.html
    )doc");

}  // namespace tensorflow
