/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_io/core/kernels/avro/atds/errors.h"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
namespace atds {

namespace {
constexpr char kSupportedDTypeMessage[] =
    "Only DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_STRING, and DT_BOOL are "
    "supported.";
}  // namespace

void TypeNotSupportedAbort(DataType dtype) {
  LOG(ERROR) << "Data type " << DataTypeString(dtype) << " is not supported. "
             << kSupportedDTypeMessage;
  std::abort();
}

void SparseIndicesTypeNotSupportedAbort(avro::Type indices_type) {
  LOG(ERROR) << "Sparse indices type " << avro::toString(indices_type)
             << " is not supported. Only AVRO_INT and AVRO_LONG are supported";
  std::abort();
}

Status TypeNotSupportedError(DataType dtype) {
  return errors::InvalidArgument(
      strings::StrCat("Data type ", DataTypeString(dtype), " is not supported.",
                      kSupportedDTypeMessage));
}

Status SparseArraysNotEqualError(const std::vector<size_t>& decoded_numbers,
                                 const std::vector<size_t>& feature_index) {
  size_t rank = decoded_numbers.size() - 1;
  string array_names = "[";
  string decoded_values = "[";
  for (size_t i = 0; i <= rank; i++) {
    if (i > 0) {
      strings::StrAppend(&array_names, ", ");
      strings::StrAppend(&decoded_values, ", ");
    }
    strings::StrAppend(&decoded_values, decoded_numbers[i]);

    size_t index = feature_index[i];
    if (index == rank) {
      strings::StrAppend(&array_names, "values");
    } else {
      strings::StrAppend(&array_names, "indices", index);
    }
  }
  strings::StrAppend(&array_names, "]");
  strings::StrAppend(&decoded_values, "]");

  return errors::InvalidArgument(strings::StrCat(
      "Numbers of decoded value in indice and values array are different. ",
      "Numbers of decoded value in ", array_names, " arrays are ",
      decoded_values));
}

Status ShapeError(size_t number, int dim, const PartialTensorShape& shape) {
  return errors::InvalidArgument(strings::StrCat(
      "Number of decoded value ", number,
      " does not match the expected dimension size ", shape.dim_size(dim),
      " at the ", dim + 1, "th dimension in user defined shape ",
      shape.DebugString()));
}

Status NullValueError() {
  return errors::InvalidArgument("Feature value is null.");
}

Status FeatureDecodeError(const string& feature_name, const string& reason) {
  return errors::InvalidArgument(strings::StrCat(
      "Failed to decode feature ", feature_name, ". Reason: ", reason));
}

Status ATDSNotRecordError(const string& type, const string& schema) {
  return errors::InvalidArgument(
      strings::StrCat("ATDS schema is expected to be an Avro Record but found ",
                      type, ". Invalid schema found: ", schema));
}

Status FeatureNotFoundError(const string& feature_name, const string& schema) {
  return errors::InvalidArgument(strings::StrCat(
      "User defined feature '", feature_name,
      "' cannot be found in the input data.", " Input data schema: ", schema));
}

Status InvalidUnionTypeError(const string& feature_name, const string& schema) {
  return errors::InvalidArgument(
      strings::StrCat("Feature '", feature_name, "' has invalid union schema. ",
                      "A feature can only be an union of itself or an union of "
                      "'null' type and itself.",
                      "Invalid union schema found: ", schema));
}

Status MissingValuesColumnError(const string& schema) {
  return errors::InvalidArgument(strings::StrCat(
      "Sparse schema is missing values column. Input data schema: ", schema));
}

Status NonContiguousIndicesError(const string& schema) {
  return errors::InvalidArgument(strings::StrCat(
      "Sparse schema indices should be contiguous (indices0, indices1, ...). ",
      "Input data schema: ", schema));
}

Status ExtraFieldError(const string& schema) {
  return errors::InvalidArgument(
      strings::StrCat("Sparse schema can only contain 'indices' columns and a "
                      "'values' column. ",
                      "Input data schema: ", schema));
}

Status UnsupportedSparseIndicesTypeError(const string& feature_name,
                                         const string& schema) {
  return errors::InvalidArgument(strings::StrCat(
      "Unsupported indices type found in feature '", feature_name, "'. ",
      "Sparse tensor indices must be a non-nullable array of non-nullable int "
      "or long. "
      "Invalid schema found: ",
      schema));
}

Status UnsupportedValueTypeError(const string& feature_name,
                                 const string& schema) {
  return errors::InvalidArgument(strings::StrCat(
      "Unsupported value type found in feature '", feature_name, "'. ",
      "Tensor value must be a non-nullable array of non-nullable int, long, "
      "float, double, boolean, bytes, or string. "
      "Invalid schema found: ",
      schema));
}

Status SchemaValueTypeMismatch(const string& feature_name, avro::Type avro_type,
                               DataType metadata_type, const string& schema) {
  return errors::InvalidArgument(strings::StrCat(
      "Schema value type and metadata type mismatch in feature '", feature_name,
      "'. ", "Avro schema data type: ", avro::toString(avro_type),
      ", metadata type: ", DataTypeString(metadata_type),
      ". Invalid schema found: ", schema));
}

Status InvalidDenseFeatureSchema(const string& feature_name,
                                 const string& schema) {
  return errors::InvalidArgument(
      strings::StrCat("Dense feature '", feature_name,
                      "' must be non-nullable nested arrays only. ",
                      "Invalid schema found: ", schema));
}

Status InvalidVarlenFeatureSchema(const string& feature_name,
                                  const string& schema) {
  return errors::InvalidArgument(
      strings::StrCat("Varlen feature '", feature_name,
                      "' must be non-nullable nested arrays only. ",
                      "Invalid schema found: ", schema));
}

Status FeatureRankMismatch(const string& feature_name, size_t avro_rank,
                           size_t metadata_rank, const string& schema) {
  return errors::InvalidArgument(strings::StrCat(
      "Mismatch between avro schema rank and metadata rank in feature '",
      feature_name, "'. ", "Avro schema rank: ", std::to_string(avro_rank),
      ", metadata rank: ", std::to_string(metadata_rank), ". ",
      "Invalid schema found: ", schema));
}

Status VariedSchemaNotSupportedError(const string& expected_schema,
                                     const string& filename,
                                     const string& varied_schema,
                                     const string& next_filename) {
  return errors::InvalidArgument(strings::StrCat(
      "Avro schema should be consistent for all input files.",
      " Schema in file ", filename, " varies from the schema in file ",
      next_filename, "\n", expected_schema, "\n != \n", varied_schema));
}

}  // namespace atds
}  // namespace tensorflow
