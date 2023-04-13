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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_SPARSE_FEATURE_DECODER_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_SPARSE_FEATURE_DECODER_H_

#include "api/Decoder.hh"
#include "api/Node.hh"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow_io/core/kernels/avro/atds/avro_decoder_template.h"
#include "tensorflow_io/core/kernels/avro/atds/decoder_base.h"
#include "tensorflow_io/core/kernels/avro/atds/errors.h"
#include "tensorflow_io/core/kernels/avro/atds/sparse_feature_internal_decoder.h"

namespace tensorflow {
namespace atds {

namespace sparse {

struct Metadata {
  Metadata(FeatureType type, const string& name, DataType dtype,
           const PartialTensorShape& shape, size_t indices_index,
           size_t values_index)
      : type(type),
        name(name),
        dtype(dtype),
        shape(shape),
        indices_index(indices_index),
        values_index(values_index) {}

  FeatureType type;
  string name;
  DataType dtype;
  PartialTensorShape shape;

  size_t indices_index;
  size_t values_index;
};

template <typename T>
class FeatureDecoder : public DecoderBase {
 public:
  explicit FeatureDecoder(const Metadata& metadata,
                          const std::vector<size_t>& decoding_order,
                          const std::vector<avro::Type>& indices_type)
      : metadata_(metadata),
        rank_(metadata.shape.dims()),
        decoding_order_(decoding_order),
        long_indices_decoder_(metadata.indices_index, rank_),
        int_indices_decoder_(metadata.indices_index, rank_),
        values_decoder_(metadata.values_index) {
    auto num_decoders = decoding_order.size();
    decoders_.reserve(num_decoders);
    for (size_t i = 0; i < num_decoders; i++) {
      auto index = decoding_order[i];
      if (index == rank_) {
        decoders_.emplace_back(&values_decoder_);
      } else if (indices_type[index] == avro::AVRO_LONG) {
        decoders_.emplace_back(&long_indices_decoder_);
      } else if (indices_type[index] == avro::AVRO_INT) {
        decoders_.emplace_back(&int_indices_decoder_);
      } else {
        SparseIndicesTypeNotSupportedAbort(indices_type[index]);
      }
    }
  }

  Status operator()(avro::DecoderPtr& decoder,
                    std::vector<Tensor>& dense_tensors,
                    sparse::ValueBuffer& buffer,
                    std::vector<avro::GenericDatum>& skipped_data,
                    size_t offset) {
    size_t num_decoders = decoders_.size();
    std::vector<size_t> decoded_numbers(num_decoders, 0);
    size_t indices_index = metadata_.indices_index;
    size_t indices_start = buffer.indices[indices_index].size();
    for (size_t i = 0; i < num_decoders; i++) {
      decoded_numbers[i] = decoders_[i]->Decode(
          decoder, buffer, decoding_order_[i], indices_start);
    }

    if (TF_PREDICT_FALSE(!std::all_of(
            decoded_numbers.cbegin(), decoded_numbers.cend(),
            [d = decoded_numbers[0]](size_t n) { return n == d; }))) {
      return SparseArraysNotEqualError(decoded_numbers, decoding_order_);
    }

    // Rank after batching equals to the number of decoders.
    FillBatchIndices(buffer.indices[indices_index], indices_start,
                     static_cast<long>(offset), num_decoders);

    auto& num_of_elements = buffer.num_of_elements[indices_index];
    size_t total_num_elements = decoded_numbers[0];
    if (!num_of_elements.empty()) {
      total_num_elements += num_of_elements.back();
    }
    num_of_elements.push_back(total_num_elements);
    return OkStatus();
  }

 private:
  void FillBatchIndices(std::vector<long>& v, size_t indices_start,
                        long batch_offset, size_t rank_after_batch) {
    size_t end = v.size();
    for (size_t i = indices_start; i < end; i += rank_after_batch) {
      v[i] = batch_offset;
    }
  }

  const Metadata& metadata_;
  const size_t rank_;
  const std::vector<size_t> decoding_order_;
  IndicesDecoder<long> long_indices_decoder_;
  IndicesDecoder<int> int_indices_decoder_;
  ValuesDecoder<T> values_decoder_;
  std::vector<InternalDecoder*> decoders_;  // not owned.
};

}  // namespace sparse

template <>
inline std::unique_ptr<DecoderBase> CreateFeatureDecoder(
    const avro::NodePtr& node, const sparse::Metadata& metadata) {
  size_t rank = static_cast<size_t>(metadata.shape.dims());
  std::vector<size_t> decoding_order(rank + 1);
  std::vector<avro::Type> indices_types(rank);

  for (size_t d = 0; d < rank; d++) {
    auto indice_key = "indices" + std::to_string(d);
    size_t indice_pos;
    node->nameIndex(indice_key, indice_pos);
    decoding_order[indice_pos] = d;
    indices_types[d] = node->leafAt(indice_pos)->leafAt(0)->type();
  }

  size_t values_pos;
  node->nameIndex("values", values_pos);
  decoding_order[values_pos] = rank;

  switch (metadata.dtype) {
    case DT_INT32: {
      return std::move(std::make_unique<sparse::FeatureDecoder<int>>(
          metadata, decoding_order, indices_types));
    }
    case DT_INT64: {
      return std::move(std::make_unique<sparse::FeatureDecoder<long>>(
          metadata, decoding_order, indices_types));
    }
    case DT_FLOAT: {
      return std::move(std::make_unique<sparse::FeatureDecoder<float>>(
          metadata, decoding_order, indices_types));
    }
    case DT_DOUBLE: {
      return std::move(std::make_unique<sparse::FeatureDecoder<double>>(
          metadata, decoding_order, indices_types));
    }
    case DT_STRING: {
      return std::move(std::make_unique<sparse::FeatureDecoder<string>>(
          metadata, decoding_order, indices_types));
    }
    case DT_BOOL: {
      return std::move(std::make_unique<sparse::FeatureDecoder<bool>>(
          metadata, decoding_order, indices_types));
    }
    default: {
      TypeNotSupportedAbort(metadata.dtype);
    }
  }
  return nullptr;
}

template <>
inline Status ValidateSchema(const avro::NodePtr& node,
                             const sparse::Metadata& metadata) {
  size_t values_pos;
  // Check values column exists.
  if (!node->nameIndex("values", values_pos)) {
    std::ostringstream oss;
    node->printJson(oss, 0);
    return MissingValuesColumnError(oss.str());
  }
  // Check values column is a non-nullable array.
  auto value_leaf = node->leafAt(values_pos);
  avro::Type value_type = value_leaf->type();
  if (value_type != avro::AVRO_ARRAY) {
    std::ostringstream oss;
    node->printJson(oss, 0);
    return UnsupportedValueTypeError(metadata.name, oss.str());
  }
  avro::Type value_item_type = value_leaf->leafAt(0)->type();
  std::map<avro::Type, DataType>::const_iterator tf_type =
      avro_to_tf_datatype.find(value_item_type);
  if (tf_type == avro_to_tf_datatype.end()) {
    // Check schema data type is supported.
    std::ostringstream oss;
    node->printJson(oss, 0);
    return UnsupportedValueTypeError(metadata.name, oss.str());
  } else if (tf_type->second != metadata.dtype) {
    // Check schema data type and metadata type match.
    std::ostringstream oss;
    node->printJson(oss, 0);
    return SchemaValueTypeMismatch(metadata.name, value_item_type,
                                   metadata.dtype, oss.str());
  }
  size_t rank = static_cast<size_t>(metadata.shape.dims());
  for (size_t i = 0; i < rank; i++) {
    auto indice_key = "indices" + std::to_string(i);
    size_t indice_pos;
    // Check for contiguous "indices0", "indices1", ... "indicesN" columns
    if (!node->nameIndex(indice_key, indice_pos)) {
      std::ostringstream oss;
      node->printJson(oss, 0);
      return NonContiguousIndicesError(oss.str());
    }
    // Check each "indices" column is a non-nullable array.
    auto indice_leaf = node->leafAt(indice_pos);
    avro::Type indices_type = indice_leaf->type();
    if (indices_type != avro::AVRO_ARRAY) {
      std::ostringstream oss;
      node->printJson(oss, 0);
      return UnsupportedSparseIndicesTypeError(metadata.name, oss.str());
    }
    // Check each "indices" array consists of int or long.
    avro::Type item_type = indice_leaf->leafAt(0)->type();
    if (item_type != avro::AVRO_INT && item_type != avro::AVRO_LONG) {
      std::ostringstream oss;
      node->printJson(oss, 0);
      return UnsupportedSparseIndicesTypeError(metadata.name, oss.str());
    }
  }
  // Check schema rank and metadata rank match.
  if (node->leaves() != rank + 1) {
    std::ostringstream oss;
    node->printJson(oss, 0);
    return ExtraFieldError(oss.str());
  }
  return OkStatus();
}

}  // namespace atds
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_SPARSE_FEATURE_DECODER_H_
