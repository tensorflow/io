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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DECODER_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DECODER_H_

#include "api/Decoder.hh"
#include "api/GenericDatum.hh"
#include "api/ValidSchema.hh"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow_io/core/kernels/avro/atds/decoder_base.h"
#include "tensorflow_io/core/kernels/avro/atds/dense_feature_decoder.h"
#include "tensorflow_io/core/kernels/avro/atds/errors.h"
#include "tensorflow_io/core/kernels/avro/atds/sparse_feature_decoder.h"
#include "tensorflow_io/core/kernels/avro/atds/varlen_feature_decoder.h"

namespace tensorflow {
namespace atds {

class NullableFeatureDecoder : public DecoderBase {
 public:
  explicit NullableFeatureDecoder(std::unique_ptr<DecoderBase>& decoder,
                                  size_t non_null_index)
      : decoder_(std::move(decoder)), non_null_index_(non_null_index) {}

  Status operator()(avro::DecoderPtr& decoder,
                    std::vector<Tensor>& dense_tensors,
                    sparse::ValueBuffer& buffer,
                    std::vector<avro::GenericDatum>& skipped_data,
                    size_t offset) {
    auto index = decoder->decodeUnionIndex();
    if (index != non_null_index_) {
      return NullValueError();
    }
    return decoder_->operator()(decoder, dense_tensors, buffer, skipped_data,
                                offset);
  }

 private:
  std::unique_ptr<DecoderBase> decoder_;
  const size_t non_null_index_;
};

class ATDSDecoder {
 public:
  explicit ATDSDecoder(const std::vector<dense::Metadata>& dense_features,
                       const std::vector<sparse::Metadata>& sparse_features,
                       const std::vector<varlen::Metadata>& varlen_features)
      : dense_features_(dense_features),
        sparse_features_(sparse_features),
        varlen_features_(varlen_features) {}

  Status Initialize(const avro::ValidSchema&);

  Status DecodeATDSDatum(avro::DecoderPtr& decoder,
                         std::vector<Tensor>& dense_tensors,
                         sparse::ValueBuffer& buffer,
                         std::vector<avro::GenericDatum>& skipped_data,
                         size_t offset) {
    // LOG(INFO) << "Decode atds from offset: " << offset;
    for (size_t i = 0; i < decoders_.size(); i++) {
      Status status = decoders_[i]->operator()(decoder, dense_tensors, buffer,
                                               skipped_data, offset);
      if (TF_PREDICT_FALSE(!status.ok())) {
        return FeatureDecodeError(feature_names_[i], string(status.message()));
      }
    }
    // LOG(INFO) << "Decode atds from offset Done: " << offset;
    return OkStatus();
  }

  const std::vector<avro::GenericDatum>& GetSkippedData() {
    return skipped_data_;
  }

  const avro::ValidSchema& GetSchema() { return schema_; }

 private:
  template <typename Metadata>
  Status InitializeFeatureDecoder(const avro::ValidSchema& schema,
                                  const avro::NodePtr& root_node,
                                  const Metadata& metadata) {
    size_t pos;
    if (!root_node->nameIndex(metadata.name, pos)) {
      return FeatureNotFoundError(metadata.name, schema.toJson());
    }
    decoder_types_[pos] = metadata.type;
    feature_names_[pos] = metadata.name;

    auto& feature_node = root_node->leafAt(pos);
    if (feature_node->type() == avro::AVRO_UNION) {
      size_t non_null_index = 0;
      size_t num_union_types = feature_node->leaves();

      if (num_union_types == 2 &&
          feature_node->leafAt(0)->type() == avro::AVRO_NULL) {
        non_null_index = 1;
      }

      if (num_union_types == 1 || num_union_types == 2) {
        auto& non_null_feature_node = feature_node->leafAt(non_null_index);
        TF_RETURN_IF_ERROR(ValidateSchema(non_null_feature_node, metadata));
        std::unique_ptr<DecoderBase> decoder_base =
            CreateFeatureDecoder(non_null_feature_node, metadata);
        decoders_[pos] = std::unique_ptr<DecoderBase>(
            new NullableFeatureDecoder(decoder_base, non_null_index));
      } else {
        std::ostringstream oss;
        feature_node->printJson(oss, 0);
        return InvalidUnionTypeError(metadata.name, oss.str());
      }
    } else {
      TF_RETURN_IF_ERROR(ValidateSchema(feature_node, metadata));
      decoders_[pos] = CreateFeatureDecoder(feature_node, metadata);
    }

    return OkStatus();
  }

  const std::vector<dense::Metadata>& dense_features_;
  const std::vector<sparse::Metadata>& sparse_features_;
  const std::vector<varlen::Metadata>& varlen_features_;

  std::vector<string> feature_names_;
  std::vector<std::unique_ptr<DecoderBase>> decoders_;
  std::vector<FeatureType> decoder_types_;

  std::vector<avro::GenericDatum> skipped_data_;
  avro::ValidSchema schema_;
};

}  // namespace atds
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DECODER_H_
