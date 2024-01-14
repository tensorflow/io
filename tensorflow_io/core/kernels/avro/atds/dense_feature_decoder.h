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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DENSE_FEATURE_DECODER_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DENSE_FEATURE_DECODER_H_

#include "api/Decoder.hh"
#include "api/Node.hh"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_io/core/kernels/avro/atds/avro_decoder_template.h"
#include "tensorflow_io/core/kernels/avro/atds/decoder_base.h"
#include "tensorflow_io/core/kernels/avro/atds/errors.h"

namespace tensorflow {
namespace atds {

namespace dense {

struct Metadata {
  Metadata(FeatureType type, const string& name, DataType dtype,
           const PartialTensorShape& shape, size_t tensor_position)
      : type(type),
        name(name),
        dtype(dtype),
        shape(shape),
        tensor_position(tensor_position) {}

  FeatureType type;
  string name;
  DataType dtype;
  PartialTensorShape shape;

  size_t tensor_position;
};

template <typename T>
inline Status DecodeFixedLenArray(avro::DecoderPtr& decoder, T** buf, int rank,
                                  const PartialTensorShape& shape) {
  if (rank == 0) {
    *((*buf)++) = avro::decoder_t::Decode<T>(decoder);
    return OkStatus();
  }

  int dim = shape.dims() - rank;
  size_t size = static_cast<size_t>(shape.dim_size(dim));
  size_t number = 0;
  if (rank == 1) {
    for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
      number += m;
      if (TF_PREDICT_FALSE(number > size)) {
        return ShapeError(number, dim, shape);
      }
      for (size_t i = 0; i < m; i++) {
        *((*buf)++) = avro::decoder_t::Decode<T>(decoder);
      }
    }
    if (TF_PREDICT_FALSE(number != size)) {
      return ShapeError(number, dim, shape);
    }
    return OkStatus();
  }

  for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
    number += m;
    if (TF_PREDICT_FALSE(number > size)) {
      return ShapeError(number, dim, shape);
    }
    for (size_t i = 0; i < m; i++) {
      TF_RETURN_IF_ERROR(DecodeFixedLenArray<T>(decoder, buf, rank - 1, shape));
    }
  }
  if (TF_PREDICT_FALSE(number != size)) {
    return ShapeError(number, dim, shape);
  }
  return OkStatus();
}

// This template specification handles both byte and string.
// It assumes that avro decodeBytes and decodeString are both reading bytes into
// uint8 arrays see:
// https://github.com/apache/avro/blob/branch-1.9/lang/c%2B%2B/impl/BinaryDecoder.cc#L133
// As long as that as that assumption holds a separate bytes implementation is
// not required.
template <>
inline Status DecodeFixedLenArray(avro::DecoderPtr& decoder, tstring** buf,
                                  int rank, const PartialTensorShape& shape) {
  std::string s;
  if (rank == 0) {
    decoder->decodeString(s);
    *((*buf)++) = s;
    return OkStatus();
  }

  int dim = shape.dims() - rank;
  size_t size = static_cast<size_t>(shape.dim_size(dim));
  size_t number = 0;
  if (rank == 1) {
    for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
      number += m;
      if (TF_PREDICT_FALSE(number > size)) {
        return ShapeError(number, dim, shape);
      }
      for (size_t i = 0; i < m; i++) {
        decoder->decodeString(s);
        *((*buf)++) = s;
      }
    }
    if (TF_PREDICT_FALSE(number != size)) {
      return ShapeError(number, dim, shape);
    }
    return OkStatus();
  }

  for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
    number += m;
    if (TF_PREDICT_FALSE(number > size)) {
      return ShapeError(number, dim, shape);
    }
    for (size_t i = 0; i < m; i++) {
      TF_RETURN_IF_ERROR(
          DecodeFixedLenArray<tstring>(decoder, buf, rank - 1, shape));
    }
  }
  if (TF_PREDICT_FALSE(number != size)) {
    return ShapeError(number, dim, shape);
  }
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
    auto size = metadata_.shape.num_elements();
    auto& tensor = dense_tensors[metadata_.tensor_position];
    T* buf = reinterpret_cast<T*>(tensor.data()) + offset * size;
    return DecodeFixedLenArray<T>(decoder, &buf, rank_, metadata_.shape);
  }

 private:
  const Metadata& metadata_;
  const int rank_;
};

}  // namespace dense

template <>
inline std::unique_ptr<DecoderBase> CreateFeatureDecoder(
    const avro::NodePtr& node, const dense::Metadata& metadata) {
  switch (metadata.dtype) {
    case DT_INT32: {
      return std::move(std::make_unique<dense::FeatureDecoder<int>>(metadata));
    }
    case DT_INT64: {
      return std::move(std::make_unique<dense::FeatureDecoder<long>>(metadata));
    }
    case DT_FLOAT: {
      return std::move(
          std::make_unique<dense::FeatureDecoder<float>>(metadata));
    }
    case DT_DOUBLE: {
      return std::move(
          std::make_unique<dense::FeatureDecoder<double>>(metadata));
    }
    case DT_STRING: {
      return std::move(
          std::make_unique<dense::FeatureDecoder<tstring>>(metadata));
    }
    case DT_BOOL: {
      return std::move(std::make_unique<dense::FeatureDecoder<bool>>(metadata));
    }
    default: {
      TypeNotSupportedAbort(metadata.dtype);
    }
  }
  return nullptr;
}

template <>
inline Status ValidateSchema(const avro::NodePtr& node,
                             const dense::Metadata& metadata) {
  avro::NodePtr n = node;
  size_t avro_rank = 0;
  // Check schema consists of non-nullable nested arrays.
  while (n->leaves() != 0) {
    if (n->leaves() != 1 || n->type() != avro::AVRO_ARRAY) {
      std::ostringstream oss;
      node->printJson(oss, 0);
      return InvalidDenseFeatureSchema(metadata.name, oss.str());
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

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DENSE_FEATURE_DECODER_H_
