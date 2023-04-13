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
