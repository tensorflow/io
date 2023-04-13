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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DECOMPRESSION_HANDLER_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DECOMPRESSION_HANDLER_H_

#include <boost/crc.hpp>  // for boost::crc_32_type
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "api/Compiler.hh"
#include "api/DataFile.hh"
#include "api/Decoder.hh"
#include "api/Specific.hh"
#include "api/Stream.hh"
#include "api/ValidSchema.hh"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_io/core/kernels/avro/atds/avro_block_reader.h"

#ifdef SNAPPY_CODEC_AVAILABLE
#include <snappy.h>
#endif
namespace tensorflow {
namespace data {
class DecompressionHandler {
 public:
  DecompressionHandler() {}

  // Adapted from
  // https://github.com/apache/avro/blob/release-1.9.1/lang/c++/impl/DataFile.cc#L58
  boost::iostreams::zlib_params get_zlib_params() {
    boost::iostreams::zlib_params ret;
    ret.method = boost::iostreams::zlib::deflated;
    ret.noheader = true;
    return ret;
  }

#ifdef SNAPPY_CODEC_AVAILABLE
  avro::InputStreamPtr decompressSnappyCodec(AvroBlock& block) {
    boost::crc_32_type crc;
    std::string uncompressed;
    size_t len = block.content.size();
    const auto& compressed = block.content;
    int b1 = compressed[len - 4] & 0xFF;
    int b2 = compressed[len - 3] & 0xFF;
    int b3 = compressed[len - 2] & 0xFF;
    int b4 = compressed[len - 1] & 0xFF;

    uint32_t checksum = (b1 << 24) + (b2 << 16) + (b3 << 8) + (b4);
    if (!snappy::Uncompress(compressed.data(), len - 4, &uncompressed)) {
      throw avro::Exception(
          "Snappy Compression reported an error when decompressing");
    }
    crc.process_bytes(uncompressed.data(), uncompressed.size());
    uint32_t c = crc();
    if (checksum != c) {
      throw avro::Exception(
          boost::format("Checksum did not match for Snappy compression: "
                        "Expected: %1%, computed: %2%") %
          checksum % c);
    }
    block.content = uncompressed;
    block.byte_count = uncompressed.size();
    block.codec = avro::NULL_CODEC;
    uint8_t* dt =
        reinterpret_cast<uint8_t*>(block.content.data() + block.read_offset);
    return avro::memoryInputStream(dt,
                                   block.content.size() - block.read_offset);
  }
#endif

  avro::InputStreamPtr decompressDeflateCodec(AvroBlock& block) {
    boost::iostreams::filtering_istream stream;
    stream.push(boost::iostreams::zlib_decompressor(get_zlib_params()));
    stream.push(boost::iostreams::basic_array_source<char>(
        block.content.data(), block.content.size()));
    auto uncompressed = tstring();
    auto reader = avro::nonSeekableIstreamInputStream(stream);
    size_t n_data = 0;

    const uint8_t* data = nullptr;
    while (reader->next(&data, &n_data)) {
      uncompressed.append((const char*)data, n_data);
    }
    block.content = uncompressed;
    block.codec = avro::NULL_CODEC;
    block.byte_count = uncompressed.size();
    uint8_t* dt =
        reinterpret_cast<uint8_t*>(block.content.data() + block.read_offset);
    return avro::memoryInputStream(dt,
                                   block.content.size() - block.read_offset);
  }

  avro::InputStreamPtr decompressNullCodec(AvroBlock& block) {
    size_t offset = block.read_offset;
    uint8_t* data = reinterpret_cast<uint8_t*>(block.content.data() + offset);
    size_t size = block.content.size() - offset;
    return avro::memoryInputStream(data, size);
  }
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DECOMPRESSION_HANDLER_H_