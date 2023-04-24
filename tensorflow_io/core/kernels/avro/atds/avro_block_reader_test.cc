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

#include "tensorflow_io/core/kernels/avro/atds/avro_block_reader.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow_io/core/kernels/avro/atds/decoder_test_util.h"
//#include "tensorflow/tsl/platform/default/posix_file_system.h"

#include <fstream>
#include <ostream>

#include "api/DataFile.hh"
#include "api/Generic.hh"
#include "api/GenericDatum.hh"
#include "api/Stream.hh"

namespace tensorflow {
namespace data {

class MockRandomAccessFile : public RandomAccessFile {
 public:
  explicit MockRandomAccessFile(char* content, size_t len)
      : content_(content), len_(len) {}

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    size_t bytes_to_copy = std::min(n, len_ - static_cast<size_t>(offset));
    memcpy(scratch, content_ + offset, bytes_to_copy);
    *result = StringPiece(scratch, bytes_to_copy);
    if (bytes_to_copy == n) {
      return OkStatus();
    }
    return Status(tensorflow::error::Code::OUT_OF_RANGE, "eof");
  }

 private:
  const char* content_;
  size_t len_;
};

TEST(FileBufferInputStreamTest, SINGLE_BUFFER) {
  char content[8];
  for (size_t i = 0; i < 8; i++) {
    content[i] = '0' + i;
  }
  std::unique_ptr<tensorflow::RandomAccessFile> raf =
      absl::make_unique<MockRandomAccessFile>(content, 8);
  int64 buffer_size = 8;
  FileBufferInputStream stream(raf.get(), buffer_size);
  const uint8_t* data;
  size_t len = 4;
  ASSERT_TRUE(stream.next(&data, &len));
  ASSERT_EQ(4, len);
  ASSERT_EQ(4, stream.byteCount());
  tensorflow::atds::AssertValueEqual("0123", (char*)data, len);

  stream.skip(1);
  len = 3;
  stream.next(&data, &len);
  ASSERT_EQ(3, len);
  ASSERT_EQ(8, stream.byteCount());
  tensorflow::atds::AssertValueEqual("567", (char*)data, len);

  stream.backup(5);
  len = 3;
  stream.next(&data, &len);
  ASSERT_EQ(3, len);
  ASSERT_EQ(6, stream.byteCount());
  tensorflow::atds::AssertValueEqual("345", (char*)data, len);
}

TEST(FileBufferInputStreamTest, READ_PAST_BUFFER) {
  char content[16];
  for (size_t i = 0; i < 16; i++) {
    content[i] = 'a' + i;
  }
  std::unique_ptr<tensorflow::RandomAccessFile> raf =
      absl::make_unique<MockRandomAccessFile>(content, 16);
  int64 buffer_size = 8;
  FileBufferInputStream stream(raf.get(), buffer_size);
  const uint8_t* data;
  size_t len = 3;
  ASSERT_TRUE(stream.next(&data, &len));
  ASSERT_EQ(3, len);
  ASSERT_EQ(3, stream.byteCount());
  tensorflow::atds::AssertValueEqual("abc", (char*)data, len);

  len = 7;
  stream.next(&data, &len);
  ASSERT_EQ(5, len);
  ASSERT_EQ(8, stream.byteCount());
  tensorflow::atds::AssertValueEqual("defgh", (char*)data, len);

  len = 4;
  stream.next(&data, &len);
  ASSERT_EQ(4, len);
  ASSERT_EQ(12, stream.byteCount());
  tensorflow::atds::AssertValueEqual("ijkl", (char*)data, len);
}

TEST(FileBufferInputStreamTest, SKIP_PAST_BUFFER) {
  char content[16];
  for (size_t i = 0; i < 16; i++) {
    content[i] = 'a' + i;
  }
  std::unique_ptr<tensorflow::RandomAccessFile> raf =
      absl::make_unique<MockRandomAccessFile>(content, 16);
  int64 buffer_size = 8;
  FileBufferInputStream stream(raf.get(), buffer_size);
  const uint8_t* data;
  size_t len = 3;
  ASSERT_TRUE(stream.next(&data, &len));
  ASSERT_EQ(3, len);
  ASSERT_EQ(3, stream.byteCount());
  tensorflow::atds::AssertValueEqual("abc", (char*)data, len);

  stream.skip(7);
  ASSERT_EQ(10, stream.byteCount());

  len = 4;
  stream.next(&data, &len);
  ASSERT_EQ(4, len);
  ASSERT_EQ(14, stream.byteCount());
  tensorflow::atds::AssertValueEqual("klmn", (char*)data, len);
}

static constexpr size_t OS_BUFFER_SIZE = 1024;

class StringOutputStream : public avro::OutputStream {
 public:
  StringOutputStream(string* buf) : buf_(buf), pos_(0) {}

  bool next(uint8_t** data, size_t* len) {
    size_t capacity = buf_->capacity();
    if (pos_ == capacity) {
      buf_->resize(capacity + OS_BUFFER_SIZE);
    }
    *data =
        reinterpret_cast<uint8_t*>(const_cast<char*>(&(buf_->c_str())[pos_]));
    size_t new_capacity = buf_->capacity();
    *len = new_capacity - pos_;
    pos_ = new_capacity;
    return true;
  }

  void backup(size_t len) { pos_ -= len; }

  uint64_t byteCount() const { return pos_; }

  void flush() {}

 private:
  string* buf_;
  size_t pos_;
};

static constexpr int64 BUFFER_SIZE = 1024;

Status AvroBlockReaderTest(char* content, size_t byte_count) {
  std::unique_ptr<tensorflow::RandomAccessFile> raf =
      absl::make_unique<MockRandomAccessFile>(content, byte_count);
  std::unique_ptr<AvroBlockReader> reader =
      absl::make_unique<AvroBlockReader>(raf.get(), BUFFER_SIZE);
  AvroBlock blk;
  return reader->ReadBlock(blk);
}

template <typename T>
void AvroBlockReaderTest(char* expected_content, int64_t object_count,
                         size_t expected_byte_count, avro::ValidSchema& schema,
                         const std::vector<T>& data) {
  string buf;
  std::unique_ptr<avro::OutputStream> os =
      absl::make_unique<StringOutputStream>(&buf);
  avro::DataFileWriter<T> writer(std::move(os), schema);
  for (T datum : data) {
    writer.write(datum);
  }
  writer.flush();
  writer.close();

  std::unique_ptr<tensorflow::RandomAccessFile> raf =
      absl::make_unique<MockRandomAccessFile>(const_cast<char*>(buf.c_str()),
                                              buf.capacity());
  std::unique_ptr<AvroBlockReader> reader =
      absl::make_unique<AvroBlockReader>(raf.get(), BUFFER_SIZE);
  tensorflow::atds::AssertValueEqual(schema, reader->GetSchema());
  AvroBlock blk;
  Status status = reader->ReadBlock(blk);
  ASSERT_TRUE(status.ok());
  tensorflow::atds::AssertValueEqual(avro::NULL_CODEC, blk.codec);
  tensorflow::atds::AssertValueEqual(object_count, blk.object_count);
  tensorflow::atds::AssertValueEqual(expected_byte_count, blk.byte_count);
  tensorflow::atds::AssertValueEqual(expected_content, blk.content.c_str(),
                                     blk.byte_count);
}

/*
 * These bytes assume the Avro file format specified here:
 * https://avro.apache.org/docs/1.9.1/spec.html#Object+Container+Files Bytes
 * were manually generated via:
 *   1. Writing schema to a file (schema.avsc):
 *      {
 *        "type" : "record",
 *        "name" : "row",
 *        "fields" : [
 *          {
 *            "name": "dense_1d",
 *            "type": {
 *              "type": "array",
 *              "items": "int"
 *            }
 *          },
 *          {
 *            "name": "dense_2d",
 *            "type": {
 *              "type": "array",
 *              "items": {
 *                "type": "array",
 *                "items": "int"
 *              }
 *            }
 *          }
 *        ]
 *      }
 *   2. Writing test data to a file (test.json):
 *     {
 *       "dense_1d": [1, 2, 3],
 *       "dense_2d": [[4, 5], [6, 7]]
 *     }
 *   3. Converting json to avro:
 *     dali avro fromjson test.json --schema-file schema.avsc > test.avro
 *
 * If avro file format changes, this byte array will need to be regenerated, and
 * test cases modified to change different byte locations in the array.
 */
static constexpr size_t BYTEARRAY_SIZE = 268;
static constexpr char WELLFORMED_CONTENT[] = {
    0x4f,
    0x62,
    0x6a,
    0x01,
    0x04,
    0x16,
    0x61,
    0x76,
    0x72,
    0x6f,
    0x2e,
    0x73,
    0x63,
    0x68,
    0x65,
    0x6d,  // Obj...avro.schem
    0x61,
    static_cast<char>(0xec),
    0x02,
    0x7b,
    0x22,
    0x74,
    0x79,
    0x70,
    0x65,
    0x22,
    0x3a,
    0x22,
    0x72,
    0x65,
    0x63,
    0x6f,  // a..{"type":"reco
    0x72,
    0x64,
    0x22,
    0x2c,
    0x22,
    0x6e,
    0x61,
    0x6d,
    0x65,
    0x22,
    0x3a,
    0x22,
    0x72,
    0x6f,
    0x77,
    0x22,  // rd","name":"row"
    0x2c,
    0x22,
    0x66,
    0x69,
    0x65,
    0x6c,
    0x64,
    0x73,
    0x22,
    0x3a,
    0x5b,
    0x7b,
    0x22,
    0x6e,
    0x61,
    0x6d,  // ,"fields":[{"nam
    0x65,
    0x22,
    0x3a,
    0x22,
    0x64,
    0x65,
    0x6e,
    0x73,
    0x65,
    0x5f,
    0x31,
    0x64,
    0x22,
    0x2c,
    0x22,
    0x74,  // e":"dense_1d","t
    0x79,
    0x70,
    0x65,
    0x22,
    0x3a,
    0x7b,
    0x22,
    0x74,
    0x79,
    0x70,
    0x65,
    0x22,
    0x3a,
    0x22,
    0x61,
    0x72,  // ype":{"type":"ar
    0x72,
    0x61,
    0x79,
    0x22,
    0x2c,
    0x22,
    0x69,
    0x74,
    0x65,
    0x6d,
    0x73,
    0x22,
    0x3a,
    0x22,
    0x69,
    0x6e,  // ray","items":"in
    0x74,
    0x22,
    0x7d,
    0x7d,
    0x2c,
    0x7b,
    0x22,
    0x6e,
    0x61,
    0x6d,
    0x65,
    0x22,
    0x3a,
    0x22,
    0x64,
    0x65,  // t"}},{"name":"de
    0x6e,
    0x73,
    0x65,
    0x5f,
    0x32,
    0x64,
    0x22,
    0x2c,
    0x22,
    0x74,
    0x79,
    0x70,
    0x65,
    0x22,
    0x3a,
    0x7b,  // nse_2d","type":{
    0x22,
    0x74,
    0x79,
    0x70,
    0x65,
    0x22,
    0x3a,
    0x22,
    0x61,
    0x72,
    0x72,
    0x61,
    0x79,
    0x22,
    0x2c,
    0x22,  // "type":"array","
    0x69,
    0x74,
    0x65,
    0x6d,
    0x73,
    0x22,
    0x3a,
    0x7b,
    0x22,
    0x74,
    0x79,
    0x70,
    0x65,
    0x22,
    0x3a,
    0x22,  // items":{"type":"
    0x61,
    0x72,
    0x72,
    0x61,
    0x79,
    0x22,
    0x2c,
    0x22,
    0x69,
    0x74,
    0x65,
    0x6d,
    0x73,
    0x22,
    0x3a,
    0x22,  // array","items":"
    0x69,
    0x6e,
    0x74,
    0x22,
    0x7d,
    0x7d,
    0x7d,
    0x5d,
    0x7d,
    0x14,
    0x61,
    0x76,
    0x72,
    0x6f,
    0x2e,
    0x63,  // int"}}}]}.avro.c
    0x6f,
    0x64,
    0x65,
    0x63,
    0x08,
    0x6e,
    0x75,
    0x6c,
    0x6c,
    0x00,
    static_cast<char>(0xe1),
    0x26,
    0x18,
    0x0e,
    static_cast<char>(0xc9),
    static_cast<char>(0xbe),  // odec.null..&....
    0x5a,
    static_cast<char>(0x8c),
    0x5f,
    static_cast<char>(0xe0),
    static_cast<char>(0xcd),
    0x5c,
    0x62,
    static_cast<char>(0xc2),
    0x3f,
    0x05,
    0x02,
    0x1e,
    0x06,
    0x02,
    0x04,
    0x06,  // Z._..\b.?.......
    0x00,
    0x04,
    0x04,
    0x08,
    0x0a,
    0x00,
    0x04,
    0x0c,
    0x0e,
    0x00,
    0x00,
    static_cast<char>(0xe1),
    0x26,
    0x18,
    0x0e,
    static_cast<char>(0xc9),  // ............&...
    static_cast<char>(0xbe),
    0x5a,
    static_cast<char>(0x8c),
    0x5f,
    static_cast<char>(0xe0),
    static_cast<char>(0xcd),
    0x5c,
    0x62,
    static_cast<char>(0xc2),
    0x3f,
    0x05,
    0x0a  // .Z._..\b.?..
};

TEST(AvroBlockReaderTest, MALFORMED_MAGIC) {
  char malformed_magic[BYTEARRAY_SIZE];
  memcpy(malformed_magic, WELLFORMED_CONTENT, BYTEARRAY_SIZE);
  malformed_magic[2] = 0x6b;  // Fill third byte with random character
  avro::Exception expected_exception("No exception thrown");
  try {
    AvroBlockReaderTest(malformed_magic, BYTEARRAY_SIZE);
  } catch (avro::Exception e) {
    expected_exception = e;
  }
  ASSERT_STREQ("Invalid data file. Magic does not match.",
               expected_exception.what());
}

TEST(AvroBlockReaderTest, MISSING_SCHEMA) {
  char missing_schema[BYTEARRAY_SIZE];
  memcpy(missing_schema, WELLFORMED_CONTENT, BYTEARRAY_SIZE);
  missing_schema[6] = 0x62;  // Replace "avro.schema" with "bvro.schema"
  avro::Exception expected_exception("No exception thrown");
  try {
    AvroBlockReaderTest(missing_schema, BYTEARRAY_SIZE);
  } catch (avro::Exception e) {
    expected_exception = e;
  }
  ASSERT_STREQ("No schema in metadata", expected_exception.what());
}

TEST(AvroBlockReaderTest, UNSUPPORTED_CODEC) {
  char unsupported_codec[BYTEARRAY_SIZE];
  memcpy(unsupported_codec, WELLFORMED_CONTENT, BYTEARRAY_SIZE);
  unsupported_codec[213] = 0x6f;  // Change codec from "null" to "oull"
  avro::Exception expected_exception("No exception thrown");
  try {
    AvroBlockReaderTest(unsupported_codec, BYTEARRAY_SIZE);
  } catch (avro::Exception e) {
    expected_exception = e;
  }
  ASSERT_STREQ("Unknown codec in data file: oull", expected_exception.what());
}

TEST(AvroBlockReaderTest, SYNC_MARKER_MISMATCH) {
  char sync_marker_mismatch[BYTEARRAY_SIZE];
  memcpy(sync_marker_mismatch, WELLFORMED_CONTENT, BYTEARRAY_SIZE);
  sync_marker_mismatch[218] =
      0xe2;  // Change second byte of sync marker from 0xe1 to 0xe2
  Status status = AvroBlockReaderTest(sync_marker_mismatch, BYTEARRAY_SIZE);
  ASSERT_EQ(error::Code::DATA_LOSS, status.code());
  ASSERT_STREQ("Avro sync marker mismatch.", status.error_message().c_str());
}

TEST(AvroBlockReaderTest, BYTE_COUNT_EOF) {
  char byte_count_eof[BYTEARRAY_SIZE];
  memcpy(byte_count_eof, WELLFORMED_CONTENT, BYTEARRAY_SIZE);
  byte_count_eof[235] = 0x6e;  // Change byte count from 0x1e (15) to 0x6e (55)
  Status status = AvroBlockReaderTest(byte_count_eof, BYTEARRAY_SIZE);
  ASSERT_EQ(error::Code::OUT_OF_RANGE, status.code());
  ASSERT_STREQ("eof", status.error_message().c_str());
}

TEST(AvroBlockReaderTest, DENSE_2D) {
  string feature_name = "dense_2d";
  tensorflow::atds::ATDSSchemaBuilder schema_builder =
      tensorflow::atds::ATDSSchemaBuilder();
  schema_builder.AddDenseFeature(feature_name, DT_INT32, 2);
  avro::ValidSchema schema = schema_builder.BuildVaildSchema();
  avro::GenericDatum datum(schema);
  tensorflow::atds::AddDenseValue<int>(datum, feature_name, {{1, 2}, {3, 4}});
  avro::OutputStreamPtr out_stream =
      tensorflow::atds::EncodeAvroGenericDatum(datum);
  avro::InputStreamPtr in_stream = avro::memoryInputStream(*out_stream);
  const uint8_t* expected_content;
  size_t expected_len;
  in_stream->next(&expected_content, &expected_len);
  AvroBlockReaderTest<avro::GenericDatum>((char*)expected_content, 1,
                                          expected_len, schema, {datum});
}

TEST(AvroBlockReaderTest, SPARSE_2D) {
  string feature_name = "sparse_2d";
  tensorflow::atds::ATDSSchemaBuilder schema_builder =
      tensorflow::atds::ATDSSchemaBuilder();
  schema_builder.AddSparseFeature(feature_name, DT_INT64, 2);
  avro::ValidSchema schema = schema_builder.BuildVaildSchema();
  avro::GenericDatum datum1(schema);
  avro::GenericDatum datum2(schema);
  tensorflow::atds::AddSparseValue<int64_t>(datum1, feature_name,
                                            {{1, 2}, {3, 4}}, {5, 6});
  tensorflow::atds::AddSparseValue<int64_t>(datum2, feature_name,
                                            {{7, 8}, {9, 10}}, {11, 12});
  std::vector<avro::GenericDatum> records = {datum1, datum2};
  avro::OutputStreamPtr out_stream =
      tensorflow::atds::EncodeAvroGenericData(records);
  avro::InputStreamPtr in_stream = avro::memoryInputStream(*out_stream);
  const uint8_t* expected_content;
  size_t expected_len;
  in_stream->next(&expected_content, &expected_len);
  AvroBlockReaderTest<avro::GenericDatum>(
      (char*)expected_content, 2, expected_len, schema, {datum1, datum2});
}

}  // namespace data
}  // namespace tensorflow
