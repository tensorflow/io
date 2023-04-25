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

#include "tensorflow_io/core/kernels/avro/atds/atds_decoder.h"

#include "api/Generic.hh"
#include "api/Specific.hh"
#include "tensorflow_io/core/kernels/avro/atds/dense_feature_decoder.h"
#include "tensorflow_io/core/kernels/avro/atds/errors.h"
#include "tensorflow_io/core/kernels/avro/atds/opaque_contextual_feature_decoder.h"
#include "tensorflow_io/core/kernels/avro/atds/sparse_feature_decoder.h"
#include "tensorflow_io/core/kernels/avro/atds/varlen_feature_decoder.h"

namespace tensorflow {
namespace atds {

Status ATDSDecoder::Initialize(const avro::ValidSchema& schema) {
  auto& root_node = schema.root();
  if (root_node->type() != avro::AVRO_RECORD) {
    return ATDSNotRecordError(avro::toString(root_node->type()),
                              schema.toJson());
  }

  size_t num_of_columns = root_node->leaves();
  feature_names_.resize(num_of_columns, "");
  decoder_types_.resize(num_of_columns, FeatureType::opaque_contextual);
  decoders_.resize(num_of_columns);

  for (size_t i = 0; i < dense_features_.size(); i++) {
    TF_RETURN_IF_ERROR(
        InitializeFeatureDecoder(schema, root_node, dense_features_[i]));
  }

  for (size_t i = 0; i < sparse_features_.size(); i++) {
    TF_RETURN_IF_ERROR(
        InitializeFeatureDecoder(schema, root_node, sparse_features_[i]));
  }

  for (size_t i = 0; i < varlen_features_.size(); i++) {
    TF_RETURN_IF_ERROR(
        InitializeFeatureDecoder(schema, root_node, varlen_features_[i]));
  }

  size_t opaque_contextual_index = 0;
  for (size_t i = 0; i < num_of_columns; i++) {
    if (decoder_types_[i] == FeatureType::opaque_contextual) {
      decoders_[i] = std::unique_ptr<DecoderBase>(
          new opaque_contextual::FeatureDecoder(opaque_contextual_index++));

      auto& opaque_contextual_node = root_node->leafAt(i);
      skipped_data_.emplace_back(opaque_contextual_node);
      if (opaque_contextual_node->hasName()) {
        feature_names_[i] = root_node->leafAt(i)->name();
        LOG(WARNING) << "Column '" << feature_names_[i] << "' from input data"
                     << " is not used. Cost of parsing an unused column is "
                        "prohibitive!! "
                     << "Consider dropping it to improve I/O performance.";
      }
    }
  }

  // Decoder requires unvaried schema in all input files.
  // Copy the schema to validate other input files.
  schema_ = schema;

  return OkStatus();
}

}  // namespace atds
}  // namespace tensorflow
