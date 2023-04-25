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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_AVRO_BLOCK_READER_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_AVRO_BLOCK_READER_H_

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zlib.hpp>

#include "api/Compiler.hh"
#include "api/DataFile.hh"
#include "api/Decoder.hh"
#include "api/Specific.hh"
#include "api/Stream.hh"
#include "api/ValidSchema.hh"
#include "tensorflow/core/lib/io/random_inputstream.h"

namespace tensorflow {
namespace data {

struct AvroBlock {
  int64_t object_count;
  int64_t num_to_decode;
  int64_t num_decoded;
  int64_t byte_count;
  int64_t counts;
  tstring content;
  avro::Codec codec;
  size_t read_offset;
};

class FileBufferInputStream : public avro::InputStream {
 public:
  FileBufferInputStream(tensorflow::RandomAccessFile* file, int64 buffer_size)
      : reader_(nullptr),
        limit_(0),
        pos_(0),
        count_(0),
        skip_(0),
        buffer_size_(buffer_size) {
    reader_ = absl::make_unique<io::RandomAccessInputStream>(file);
  }

  bool next(const uint8_t** data, size_t* len) override {
    while (pos_ == limit_) {
      if (skip_ > 0) {
        reader_->SkipNBytes(static_cast<int64>(skip_));
        skip_ = 0;
      }

      buf_.clear();
      Status status = reader_->ReadNBytes(buffer_size_, &buf_);
      pos_ = 0;
      limit_ = buf_.size();
      if (limit_ == 0 && errors::IsOutOfRange(status)) {
        return false;
      }
    }

    if (*len == 0 || pos_ + *len > limit_) {
      *len = limit_ - pos_;
    }

    *data = reinterpret_cast<uint8_t*>(buf_.data()) + pos_;
    pos_ += *len;
    count_ += *len;

    return *len != 0;
  }

  void backup(size_t len) override {
    pos_ -= len;
    count_ -= len;
  }

  void skip(size_t len) override {
    if (pos_ + len > limit_) {
      skip_ = pos_ + len - limit_;
      pos_ = limit_;
    } else {
      pos_ += len;
    }

    count_ += len;
  }

  size_t byteCount() const override { return count_; }

 private:
  std::unique_ptr<io::RandomAccessInputStream> reader_;
  size_t limit_, pos_, count_, skip_;
  const int64 buffer_size_;
  tstring buf_;
};

constexpr const char* const AVRO_SCHEMA_KEY = "avro.schema";
constexpr const char* const AVRO_CODEC_KEY = "avro.codec";
constexpr const char* const AVRO_NULL_CODEC = "null";
constexpr const char* const AVRO_DEFLATE_CODEC = "deflate";
constexpr const char* const AVRO_SNAPPY_CODEC = "snappy";

using Magic = std::array<uint8_t, 4>;
static const Magic magic = {{'O', 'b', 'j', '\x01'}};

using AvroMetadata = std::map<std::string, std::vector<uint8_t>>;

class AvroBlockReader {
 public:
  AvroBlockReader(tensorflow::RandomAccessFile* file, int64 buffer_size)
      : stream_(nullptr), decoder_(nullptr) {
    stream_ = std::make_unique<FileBufferInputStream>(file, buffer_size);
    decoder_ = avro::binaryDecoder();
    ReadHeader();
  }

  const avro::ValidSchema& GetSchema() { return data_schema_; }

  Status ReadBlock(AvroBlock& block) {
    decoder_->init(*stream_);
    const uint8_t* p = 0;
    size_t n = 0;
    if (!stream_->next(&p, &n)) {
      return errors::OutOfRange("eof");
    }
    stream_->backup(n);

    avro::decode(*decoder_, block.object_count);
    // LOG(INFO) << "block object counts = " << block.object_count;
    avro::decode(*decoder_, block.byte_count);
    // LOG(INFO) << "block bytes counts = " << block.byte_count;
    block.content.reserve(block.byte_count);

    decoder_->init(*stream_);
    int64_t remaining_bytes = block.byte_count;
    while (remaining_bytes > 0) {
      const uint8_t* data;
      size_t len = remaining_bytes;
      if (!stream_->next(&data, &len)) {
        return errors::OutOfRange("eof");
      }
      block.content.append(reinterpret_cast<const char*>(data), len);
      remaining_bytes -= len;
    }
    // LOG(INFO) << "block content = " << block.content;
    block.codec = codec_;
    block.read_offset = 0;
    block.num_decoded = 0;
    block.num_to_decode = 0;
    decoder_->init(*stream_);
    avro::DataFileSync sync_marker;
    avro::decode(*decoder_, sync_marker);
    if (sync_marker != sync_marker_) {
      return errors::DataLoss("Avro sync marker mismatch.");
    }

    return OkStatus();
  }

 private:
  void ReadHeader() {
    decoder_->init(*stream_);
    Magic m;
    avro::decode(*decoder_, m);
    if (magic != m) {
      throw avro::Exception("Invalid data file. Magic does not match.");
    }
    avro::decode(*decoder_, metadata_);
    AvroMetadata::const_iterator it = metadata_.find(AVRO_SCHEMA_KEY);
    if (it == metadata_.end()) {
      throw avro::Exception("No schema in metadata");
    }

    string schema = std::string(
        reinterpret_cast<const char*>(it->second.data()), it->second.size());
    // LOG(INFO) << schema;
    std::istringstream iss(schema);
    avro::compileJsonSchema(iss, data_schema_);

    it = metadata_.find(AVRO_CODEC_KEY);
    if (it != metadata_.end()) {
      size_t length = it->second.size();
      const char* codec = reinterpret_cast<const char*>(it->second.data());
      // LOG(INFO) << "Codec = " << std::string(codec, length);
      if (strncmp(codec, AVRO_DEFLATE_CODEC, length) == 0) {
        codec_ = avro::DEFLATE_CODEC;
      } else if (strncmp(codec, AVRO_SNAPPY_CODEC, length) == 0) {
        codec_ = avro::SNAPPY_CODEC;
      } else if (strncmp(codec, AVRO_NULL_CODEC, length) == 0) {
        codec_ = avro::NULL_CODEC;
      } else {
        throw avro::Exception("Unknown codec in data file: " +
                              std::string(codec, it->second.size()));
      }
    } else {
      codec_ = avro::NULL_CODEC;
    }

    avro::decode(*decoder_, sync_marker_);
  }

  AvroMetadata metadata_;
  avro::DataFileSync sync_marker_;
  avro::Codec codec_;

  std::unique_ptr<FileBufferInputStream> stream_;
  avro::DecoderPtr decoder_;
  avro::ValidSchema data_schema_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_AVRO_BLOCK_READER_H_
