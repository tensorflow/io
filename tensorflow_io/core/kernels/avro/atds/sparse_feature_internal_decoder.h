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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_SPARSE_FEATURE_INTERNAL_DECODER_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_SPARSE_FEATURE_INTERNAL_DECODER_H_

#include "api/Decoder.hh"
#include "tensorflow_io/core/kernels/avro/atds/avro_decoder_template.h"
#include "tensorflow_io/core/kernels/avro/atds/decoder_base.h"

namespace tensorflow {
namespace atds {
namespace sparse {

template <typename T>
inline size_t DecodeVarLenValues(avro::DecoderPtr& decoder, std::vector<T>& v) {
  size_t count = 0;
  for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
    count += m;
    for (size_t i = 0; i < m; i++) {
      v.emplace_back(avro::decoder_t::Decode<T>(decoder));
    }
  }
  return count;
}

// This template specification handles both byte and string.
// It assumes that avro decodeBytes and decodeString are both reading bytes into
// uint8 arrays see:
// https://github.com/apache/avro/blob/branch-1.9/lang/c%2B%2B/impl/BinaryDecoder.cc#L133
// As long as that as that assumption holds a separate bytes implementation is
// not required.
template <>
inline size_t DecodeVarLenValues(avro::DecoderPtr& decoder,
                                 std::vector<string>& v) {
  size_t count = 0;
  for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
    count += m;
    for (size_t i = 0; i < m; i++) {
      v.push_back("");
      decoder->decodeString(v.back());
    }
  }
  return count;
}

class InternalDecoder {
 public:
  virtual ~InternalDecoder() {}

  virtual size_t Decode(avro::DecoderPtr& decoder, ValueBuffer& buffer,
                        size_t dim, size_t indices_start) = 0;
};

template <typename T>
class ValuesDecoder : public InternalDecoder {
 public:
  explicit ValuesDecoder(size_t values_index) : values_index_(values_index) {}

  // Two size_t parameters are only used in IndicesDecoder.
  size_t Decode(avro::DecoderPtr& decoder, ValueBuffer& buffer,
                size_t not_used_1, size_t not_used_2) {
    return DecodeVarLenValues<T>(decoder,
                                 GetValueVector<T>(buffer, values_index_));
  }

 private:
  const size_t values_index_;
};

template <
    typename T,
    typename = typename std::enable_if<
        std::is_same<int, T>::value || std::is_same<long, T>::value, T>::type>
class IndicesDecoder : public InternalDecoder {
 public:
  explicit IndicesDecoder(size_t indices_index, size_t rank)
      : indices_index_(indices_index), rank_after_batch_(rank + 1) {}

  size_t Decode(avro::DecoderPtr& decoder, ValueBuffer& buffer, size_t dim,
                size_t indices_start) {
    auto& v = buffer.indices[indices_index_];
    size_t count = 0;
    size_t start = indices_start;
    auto dim_after_batch = dim + 1;
    for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
      count += m;
      size_t end = start + m * rank_after_batch_;
      if (end > v.capacity()) {
        v.reserve(2 * v.capacity());
      }
      if (end > v.size()) {
        v.resize(end);
      }
      for (size_t i = start + dim_after_batch; i < end;
           i += rank_after_batch_) {
        v[i] = static_cast<long>(avro::decoder_t::Decode<T>(decoder));
      }
      start = end;
    }
    return count;
  }

 private:
  const size_t indices_index_;
  const size_t rank_after_batch_;
};

template <>
inline size_t IndicesDecoder<long>::Decode(avro::DecoderPtr& decoder,
                                           ValueBuffer& buffer, size_t dim,
                                           size_t indices_start) {
  auto& v = buffer.indices[indices_index_];
  size_t count = 0;
  size_t start = indices_start;
  auto dim_after_batch = dim + 1;
  for (size_t m = decoder->arrayStart(); m != 0; m = decoder->arrayNext()) {
    count += m;
    size_t end = start + m * rank_after_batch_;
    if (end > v.capacity()) {
      v.reserve(2 * v.capacity());
    }
    if (end > v.size()) {
      v.resize(end);
    }
    for (size_t i = start + dim_after_batch; i < end; i += rank_after_batch_) {
      v[i] = decoder->decodeLong();
    }
    start = end;
  }
  return count;
}

}  // namespace sparse
}  // namespace atds
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_SPARSE_FEATURE_INTERNAL_DECODER_H_
