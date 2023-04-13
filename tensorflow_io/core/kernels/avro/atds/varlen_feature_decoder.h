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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_RAGGED_FEATURE_DECODER_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_RAGGED_FEATURE_DECODER_H_

#include "api/Decoder.hh"
#include "api/Node.hh"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_io/core/kernels/avro/atds/avro_decoder_template.h"
#include "tensorflow_io/core/kernels/avro/atds/decoder_base.h"
#include "tensorflow_io/core/kernels/avro/atds/errors.h"
#include "tensorflow_io/core/kernels/avro/atds/sparse_value_buffer.h"

namespace tensorflow {
namespace atds {

namespace varlen {

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

inline void FillIndicesBuffer(std::vector<long>& indices_buf,
                              std::vector<long>& current_indice) {
  for (const auto& indice_dim : current_indice) {
    indices_buf.emplace_back(indice_dim);
  }
}

template <typename T>
inline Status DecodeVarlenArray(avro::DecoderPtr& decoder,
                                std::vector<long>& indices_buf,
                                std::vector<T>& values_buf,
                                std::vector<long>& current_indice, int rank,
                                const PartialTensorShape& shape) {
  if (rank == 0) {
    FillIndicesBuffer(indices_buf, current_indice);
    values_buf.emplace_back(avro::decoder_t::Decode<T>(decoder));
    return OkStatus();
  }

  current_indice.emplace_back(0);
  int dim = shape.dims() - rank;
  int64 size = shape.dim_size(dim);
  int64 number = 0;
  if (size > 0) {
    // slow path with dimension check.
    if (rank == 1) {
      for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
        number += static_cast<int64>(m);
        if (TF_PREDICT_FALSE(number > size)) {
          return ShapeError(number, dim, shape);
        }
        for (size_t i = 0; i < m; i++) {
          FillIndicesBuffer(indices_buf, current_indice);
          values_buf.emplace_back(avro::decoder_t::Decode<T>(decoder));
          current_indice.back()++;
        }
      }
    } else {
      for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
        number += static_cast<int64>(m);
        if (TF_PREDICT_FALSE(number > size)) {
          return ShapeError(number, dim, shape);
        }
        for (size_t i = 0; i < m; i++) {
          TF_RETURN_IF_ERROR(DecodeVarlenArray<T>(decoder, indices_buf,
                                                  values_buf, current_indice,
                                                  rank - 1, shape));
          current_indice.back()++;
        }
      }
    }
    if (TF_PREDICT_FALSE(number != size)) {
      return ShapeError(number, dim, shape);
    }
  } else {
    // fast path without dimension check as the dimension can have unlimited
    // values.
    if (rank == 1) {
      for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
        for (size_t i = 0; i < m; i++) {
          FillIndicesBuffer(indices_buf, current_indice);
          values_buf.emplace_back(avro::decoder_t::Decode<T>(decoder));
          current_indice.back()++;
        }
      }
    } else {
      for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
        for (size_t i = 0; i < m; i++) {
          TF_RETURN_IF_ERROR(DecodeVarlenArray<T>(decoder, indices_buf,
                                                  values_buf, current_indice,
                                                  rank - 1, shape));
          current_indice.back()++;
        }
      }
    }
  }

  current_indice.pop_back();
  return OkStatus();
}

// This template specification handles both byte and string.
// It assumes that avro decodeBytes and decodeString are both reading bytes into
// uint8 arrays see:
// https://github.com/apache/avro/blob/branch-1.9/lang/c%2B%2B/impl/BinaryDecoder.cc#L133
// As long as that as that assumption holds a separate bytes implementation is
// not required.
template <>
inline Status DecodeVarlenArray(avro::DecoderPtr& decoder,
                                std::vector<long>& indices_buf,
                                std::vector<string>& values_buf,
                                std::vector<long>& current_indice, int rank,
                                const PartialTensorShape& shape) {
  if (rank == 0) {
    FillIndicesBuffer(indices_buf, current_indice);
    values_buf.push_back("");
    decoder->decodeString(values_buf.back());
    return OkStatus();
  }

  current_indice.emplace_back(0);
  int dim = shape.dims() - rank;
  int64 size = shape.dim_size(dim);
  int64 number = 0;
  if (size > 0) {
    // slow path with dimension check.
    if (rank == 1) {
      for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
        number += static_cast<int64>(m);
        if (TF_PREDICT_FALSE(number > size)) {
          return ShapeError(number, dim, shape);
        }
        for (size_t i = 0; i < m; i++) {
          FillIndicesBuffer(indices_buf, current_indice);
          values_buf.push_back("");
          decoder->decodeString(values_buf.back());
          current_indice.back()++;
        }
      }
    } else {
      for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
        number += static_cast<int64>(m);
        if (TF_PREDICT_FALSE(number > size)) {
          return ShapeError(number, dim, shape);
        }
        for (size_t i = 0; i < m; i++) {
          TF_RETURN_IF_ERROR(DecodeVarlenArray(decoder, indices_buf, values_buf,
                                               current_indice, rank - 1,
                                               shape));
          current_indice.back()++;
        }
      }
    }
    if (TF_PREDICT_FALSE(number != size)) {
      return ShapeError(number, dim, shape);
    }
  } else {
    // fast path without dimension check as the dimension can have unlimited
    // values.
    if (rank == 1) {
      for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
        for (size_t i = 0; i < m; i++) {
          FillIndicesBuffer(indices_buf, current_indice);
          values_buf.push_back("");
          decoder->decodeString(values_buf.back());
          current_indice.back()++;
        }
      }
    } else {
      for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
        for (size_t i = 0; i < m; i++) {
          TF_RETURN_IF_ERROR(DecodeVarlenArray(decoder, indices_buf, values_buf,
                                               current_indice, rank - 1,
                                               shape));
          current_indice.back()++;
        }
      }
    }
  }

  current_indice.pop_back();
  return OkStatus();
}

template <typename T>
class FeatureDecoder : public DecoderBase {
 public:
  explicit FeatureDecoder(const Metadata& metadata)
      : metadata_(metadata), rank_(metadata.shape.dims()) {}

  Status operator()(avro::DecoderPtr& decoder,
                    std::vector<Tensor>& dense_tensors,
                    sparse::ValueBuffer& buffer,
                    std::vector<avro::GenericDatum>& skipped_data,
                    size_t offset) {
    // declaring std::vector locally to make it thread safe
    std::vector<long> current_indices;
    current_indices.reserve(rank_ + 1);  // additional batch dim.
    current_indices.resize(1);
    current_indices[0] = offset;
    size_t indices_index = metadata_.indices_index;

    auto& indices_buf = buffer.indices[indices_index];
    auto& values_buf =
        sparse::GetValueVector<T>(buffer, metadata_.values_index);
    size_t values_buf_size = values_buf.size();
    TF_RETURN_IF_ERROR(DecodeVarlenArray<T>(decoder, indices_buf, values_buf,
                                            current_indices, rank_,
                                            metadata_.shape));
    size_t total_num_elements = values_buf.size() - values_buf_size;
    auto& num_of_elements = buffer.num_of_elements[indices_index];
    if (!num_of_elements.empty()) {
      total_num_elements += num_of_elements.back();
    }
    num_of_elements.push_back(total_num_elements);
    return OkStatus();
  }

 private:
  const Metadata& metadata_;
  const int rank_;
};

}  // namespace varlen

template <>
inline std::unique_ptr<DecoderBase> CreateFeatureDecoder(
    const avro::NodePtr& node, const varlen::Metadata& metadata) {
  switch (metadata.dtype) {
    case DT_INT32: {
      return std::move(std::make_unique<varlen::FeatureDecoder<int>>(metadata));
    }
    case DT_INT64: {
      return std::move(
          std::make_unique<varlen::FeatureDecoder<long>>(metadata));
    }
    case DT_FLOAT: {
      return std::move(
          std::make_unique<varlen::FeatureDecoder<float>>(metadata));
    }
    case DT_DOUBLE: {
      return std::move(
          std::make_unique<varlen::FeatureDecoder<double>>(metadata));
    }
    case DT_STRING: {
      return std::move(
          std::make_unique<varlen::FeatureDecoder<string>>(metadata));
    }
    case DT_BOOL: {
      return std::move(
          std::make_unique<varlen::FeatureDecoder<bool>>(metadata));
    }
    default: {
      TypeNotSupportedAbort(metadata.dtype);
    }
  }
  return nullptr;
}

template <>
inline Status ValidateSchema(const avro::NodePtr& node,
                             const varlen::Metadata& metadata) {
  avro::NodePtr n = node;
  size_t avro_rank = 0;
  // Check schema consists of non-nullable nested arrays.
  while (n->leaves() != 0) {
    if (n->leaves() != 1 || n->type() != avro::AVRO_ARRAY) {
      std::ostringstream oss;
      n->printJson(oss, 0);
      return InvalidVarlenFeatureSchema(metadata.name, oss.str());
    }
    n = n->leafAt(0);
    avro_rank++;
  }
  avro::Type avro_type = n->type();
  std::map<avro::Type, DataType>::const_iterator tf_type =
      avro_to_tf_datatype.find(avro_type);
  if (tf_type == avro_to_tf_datatype.end()) {
    // Check schema data type is supported.
    std::ostringstream oss;
    node->printJson(oss, 0);
    return UnsupportedValueTypeError(metadata.name, oss.str());
  } else if (tf_type->second != metadata.dtype) {
    // Check schema data type and metadata type match.
    std::ostringstream oss;
    node->printJson(oss, 0);
    return SchemaValueTypeMismatch(metadata.name, avro_type, metadata.dtype,
                                   oss.str());
  }
  // Check schema rank and metadata rank match.
  size_t metadata_rank = static_cast<size_t>(metadata.shape.dims());
  if (avro_rank != metadata_rank) {
    std::ostringstream oss;
    node->printJson(oss, 0);
    return FeatureRankMismatch(metadata.name, avro_rank, metadata_rank,
                               oss.str());
  }
  return OkStatus();
}

}  // namespace atds
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_RAGGED_FEATURE_DECODER_H_
