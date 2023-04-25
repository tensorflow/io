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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_AVRO_DECODER_TEMPLATE_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_AVRO_DECODER_TEMPLATE_H_

#include "api/Decoder.hh"

namespace avro {
namespace decoder_t {

template <
    typename T,
    typename = typename std::enable_if<
        std::is_same<int, T>::value || std::is_same<long, T>::value ||
            std::is_same<float, T>::value || std::is_same<double, T>::value ||
            std::is_same<bool, T>::value,
        T>::type>
inline T Decode(avro::DecoderPtr& decoder);

template <>
inline int Decode(avro::DecoderPtr& decoder) {
  return decoder->decodeInt();
}

template <>
inline long Decode(avro::DecoderPtr& decoder) {
  return decoder->decodeLong();
}

template <>
inline float Decode(avro::DecoderPtr& decoder) {
  return decoder->decodeFloat();
}

template <>
inline double Decode(avro::DecoderPtr& decoder) {
  return decoder->decodeDouble();
}

template <>
inline bool Decode(avro::DecoderPtr& decoder) {
  return decoder->decodeBool();
}

}  // namespace decoder_t
}  // namespace avro

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_AVRO_DECODER_TEMPLATE_H_
