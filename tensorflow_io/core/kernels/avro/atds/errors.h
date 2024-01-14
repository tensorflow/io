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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_ERRORS_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_ERRORS_H_

#include "api/Types.hh"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace atds {

void TypeNotSupportedAbort(DataType dtype);

void SparseIndicesTypeNotSupportedAbort(avro::Type type);

Status TypeNotSupportedError(DataType dtype);

Status SparseArraysNotEqualError(const std::vector<size_t>& decoded_numbers,
                                 const std::vector<size_t>& feature_index);

Status ShapeError(size_t number, int dim, const PartialTensorShape& shape);

Status NullValueError();

Status FeatureDecodeError(const string& feature_name, const string& reason);

Status ATDSNotRecordError(const string& type, const string& schema);

Status FeatureNotFoundError(const string& feature_name, const string& schema);

Status InvalidUnionTypeError(const string& feature_name, const string& schema);

Status MissingValuesColumnError(const string& schema);

Status NonContiguousIndicesError(const string& schema);

Status ExtraFieldError(const string& schema);

Status UnsupportedSparseIndicesTypeError(const string& feature_name,
                                         const string& schema);

Status UnsupportedValueTypeError(const string& feature_name,
                                 const string& schema);

Status SchemaValueTypeMismatch(const string& feature_name, avro::Type avro_type,
                               DataType metadata_type, const string& schema);

Status InvalidDenseFeatureSchema(const string& feature_name,
                                 const string& schema);

Status InvalidVarlenFeatureSchema(const string& feature_name,
                                  const string& schema);

Status FeatureRankMismatch(const string& feature_name, size_t avro_rank,
                           size_t metadata_rank, const string& schema);

Status VariedSchemaNotSupportedError(const string& expected_schema,
                                     const string& filename,
                                     const string& varied_schema,
                                     const string& next_filename);

}  // namespace atds
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_ERRORS_H_
