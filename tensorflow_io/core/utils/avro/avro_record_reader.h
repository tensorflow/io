/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
#ifndef TENSORFLOW_DATA_AVRO_FILE_STREAM_READER_H_
#define TENSORFLOW_DATA_AVRO_FILE_STREAM_READER_H_

#include <string>

#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow_io/avro/utils/avro_parser_tree.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "api/Stream.hh"
#include "api/DataFile.hh"
#include "api/Encoder.hh"

// mostly from here:
// https://github.com/tensorflow/tensorflow/blob/7ba3600c94bcf02e42905465e2501e56b7bd991b/tensorflow/core/lib/io/record_reader.h
// https://avro.apache.org/docs/1.7.7/api/cpp/html/index.html -- Note, in 1.8.2 the display of examples is broken
namespace tensorflow {
namespace data {

class AvroReaderOptions {
public:
  static AvroReaderOptions CreateReaderOptions() { return AvroReaderOptions(64*1024, ""); }
  int64 buffer_size;
  string reader_schema;
private:
  AvroReaderOptions(int64 buffer_size, const string& reader_schema)
    : buffer_size(buffer_size), reader_schema(reader_schema) { }
};

class AvroRecordReader {
public:
  explicit AvroRecordReader(RandomAccessFile* file, const AvroReaderOptions& options);
  virtual ~AvroRecordReader() = default;

  // Read the record at "*offset" into *record and update *offset to
  // point to the offset of the next record.  Returns OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  Status ReadRecord(uint64* offset, string* string);
private:

  std::unique_ptr<avro::GenericDatum> datum_;
  const AvroReaderOptions options_;

  // Handling avro data for decoding from file and encoding to string
  std::unique_ptr<avro::DataFileReader<avro::GenericDatum> > reader_;
  avro::EncoderPtr encoder_; // note shared ptr
  avro::ValidSchema reader_schema_;
};


class SequentialAvroRecordReader {
 public:
  // Create a reader that will return log records from "*file".
  // "*file" must remain live while this Reader is in use.
  explicit SequentialAvroRecordReader(
      RandomAccessFile* file,
      const AvroReaderOptions& options = AvroReaderOptions::CreateReaderOptions());

  virtual ~SequentialAvroRecordReader() = default;

  // Reads the next record in the file into *record. Returns OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  Status ReadRecord(string* record) {
    return underlying_.ReadRecord(&offset_, record);
  }

  // Returns the current offset in the file.
  uint64 TellOffset() { return offset_; }

  // Seek to this offset within the file and set this offset as the current
  // offset. Trying to seek backward will throw error.
  Status SeekOffset(uint64 offset) {
    if (offset < offset_)
      return errors::InvalidArgument(
          "Trying to seek offset: ", offset,
          " which is less than the current offset: ", offset_);
    offset_ = offset;
    return Status::OK();
  }

 private:
  AvroRecordReader underlying_;
  uint64 offset_ = 0;
};


}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_AVRO_FILE_STREAM_READER_H_