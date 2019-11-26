/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow_io/avro/utils/avro_record_reader.h"

#include <limits.h>

#include "api/DataFile.hh"
#include "api/Generic.hh"
#include "api/Compiler.hh"

namespace tensorflow {
namespace data {

class AvroDataInputStream : public avro::InputStream {
public:
  AvroDataInputStream(std::unique_ptr<io::BufferedInputStream> input_stream, size_t buffer_size)
    : input_stream_(std::move(input_stream)), buffer_size_(buffer_size) { }
  bool next(const uint8_t** data, size_t* len) override {

    if (*len <= 0 || *len > buffer_size_) {
      *len = buffer_size_;
    }

    if (do_seek) {
      input_stream_->Seek(pos_);
      do_seek = false;
    }

    input_stream_->ReadNBytes(*len, &chunk_);

    *data = (const uint8_t*) chunk_.data();
    *len = chunk_.size();

    pos_ += *len;

    return (*len != 0);
  }
  void backup(size_t len) override {
    do_seek = true;
    pos_ -= len;
  }
  void skip(size_t len) override {
    do_seek = true;
    pos_ += len;
  }
  size_t byteCount() const override {
    return pos_;
  }
private:
  std::unique_ptr<io::BufferedInputStream> input_stream_;
  const size_t buffer_size_;
  string chunk_;
  size_t pos_ = 0;
  bool do_seek = false;
};

AvroRecordReader::AvroRecordReader(RandomAccessFile* file,
                                   const AvroReaderOptions& options) :
    datum_(nullptr),
    options_(options),
    reader_(nullptr),
    writer_stream_(avro::memoryOutputStream()), encoder_(avro::binaryEncoder()),
    reader_stream_(avro::memoryInputStream(*writer_stream_)) {
  // TODO: Handle buffer_size = 0 in 2.0 since InputStreamInterface has Seek method
  // if (options.buffer_size > 0) {...}
  std::unique_ptr<io::BufferedInputStream> buffered_input(
    new io::BufferedInputStream(new io::RandomAccessInputStream(file), options_.buffer_size, true));
  std::unique_ptr<AvroDataInputStream> avro_input(
    new AvroDataInputStream(std::move(buffered_input), options_.buffer_size));
  // Log a warning
  string error;
  std::istringstream ss(options_.reader_schema);
  if (!avro::compileJsonSchema(ss, reader_schema_, error)) {
    // TODO: Log warning here that the writer schema is used for reading
    //return errors::InvalidArgument("Avro schema error: ", error);
    reader_.reset(new avro::DataFileReader<avro::GenericDatum>(
      std::move(avro_input)));
    datum_.reset(new avro::GenericDatum(reader_schema_));
  } else {
    reader_.reset(new avro::DataFileReader<avro::GenericDatum>(
      std::move(avro_input), reader_schema_));
    datum_.reset(new avro::GenericDatum());
  }
  encoder_->init(*writer_stream_);
}

Status AvroRecordReader::ReadRecord(uint64* offset, string* record) {
  // TODO: Wire up offset, setting, seeking etc.  note, may only be possible to sync points
  reader_->read(*datum_);
  avro::encode(*encoder_, *datum_);
  uint64_t n_data = 0;
  const uint8_t* data = nullptr;
  while (reader_stream_->next(&data, &n_data)) {
    record->append((const char*) data, n_data);
  }
  VLOG(7) << "The size of data is: " << n_data;
  record->append("This is a datum");
  //return record->empty() == 0 ? errors::OutOfRange("eof") : Status::OK();
  return errors::OutOfRange("eof");
}

SequentialAvroRecordReader::SequentialAvroRecordReader(
    RandomAccessFile* file, const AvroReaderOptions& options)
    : underlying_(file, options), offset_(0) {}

}  // namespace data
}  // namespace tensorflow