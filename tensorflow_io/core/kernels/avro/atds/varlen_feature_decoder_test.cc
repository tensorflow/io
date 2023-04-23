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

#include "tensorflow_io/core/kernels/avro/atds/varlen_feature_decoder.h"

#include "api/Decoder.hh"
#include "api/Stream.hh"
#include "api/ValidSchema.hh"
#include "tensorflow/core/platform/test.h"
#include "tensorflow_io/core/kernels/avro/atds/decoder_test_util.h"

namespace tensorflow {
namespace atds {
namespace varlen {

template <typename T, typename Type>
void VarlenDecoderTest(const T& values, DataType dtype,
                       std::initializer_list<int64> shape,
                       const std::vector<long>& expected_indices,
                       const std::vector<Type>& expected_values, long offset,
                       const avro::Type avro_type = avro::AVRO_NULL) {
  string feature_name = "feature";
  ATDSSchemaBuilder schema_builder = ATDSSchemaBuilder();
  schema_builder.AddDenseFeature(feature_name, dtype, shape.size(), avro_type);

  string schema = schema_builder.Build();
  avro::ValidSchema writer_schema = schema_builder.BuildVaildSchema();
  avro::GenericDatum atds_datum(writer_schema);
  AddDenseValue(atds_datum, feature_name, values);

  avro::OutputStreamPtr out_stream = EncodeAvroGenericDatum(atds_datum);
  avro::InputStreamPtr in_stream = avro::memoryInputStream(*out_stream);
  avro::DecoderPtr decoder = avro::binaryDecoder();
  decoder->init(*in_stream);

  std::vector<dense::Metadata> dense_features;
  std::vector<sparse::Metadata> sparse_features;
  std::vector<varlen::Metadata> varlen_features;
  size_t indices_index = 0, values_index = 0;
  PartialTensorShape tensor_shape(shape);
  varlen_features.emplace_back(FeatureType::varlen, feature_name, dtype,
                               tensor_shape, indices_index, values_index);

  ATDSDecoder atds_decoder =
      ATDSDecoder(dense_features, sparse_features, varlen_features);
  Status init_status = atds_decoder.Initialize(writer_schema);
  ASSERT_TRUE(init_status.ok());

  std::vector<avro::GenericDatum> skipped_data = atds_decoder.GetSkippedData();
  std::vector<Tensor> dense_tensors;
  sparse::ValueBuffer buffer;
  sparse::GetValuesBuffer<Type>(buffer).resize(1);
  buffer.indices.resize(1);
  buffer.num_of_elements.resize(1);
  Status decode_status =
      atds_decoder.DecodeATDSDatum(decoder, dense_tensors, buffer, skipped_data,
                                   static_cast<size_t>(offset));
  ASSERT_TRUE(decode_status.ok());

  std::vector<size_t> expected_num_elements = {expected_values.size()};

  ValidateBuffer(buffer, varlen_features[0], expected_indices, expected_values,
                 expected_num_elements);
}

template <typename T>
inline void VarlenDecoderTest(const T& values, DataType dtype,
                              std::initializer_list<int64> shape,
                              const std::vector<long>& expected_indices,
                              const std::vector<byte_array>& expected_values,
                              long offset, const avro::Type avro_type) {
  string feature_name = "feature";
  ATDSSchemaBuilder schema_builder = ATDSSchemaBuilder();
  schema_builder.AddDenseFeature(feature_name, dtype, shape.size(), avro_type);

  string schema = schema_builder.Build();
  avro::ValidSchema writer_schema = schema_builder.BuildVaildSchema();
  avro::GenericDatum atds_datum(writer_schema);
  AddDenseValue(atds_datum, feature_name, values);

  avro::OutputStreamPtr out_stream = EncodeAvroGenericDatum(atds_datum);
  avro::InputStreamPtr in_stream = avro::memoryInputStream(*out_stream);
  avro::DecoderPtr decoder = avro::binaryDecoder();
  decoder->init(*in_stream);

  std::vector<dense::Metadata> dense_features;
  std::vector<sparse::Metadata> sparse_features;
  std::vector<varlen::Metadata> varlen_features;
  size_t indices_index = 0, values_index = 0;
  PartialTensorShape tensor_shape(shape);
  varlen_features.emplace_back(FeatureType::varlen, feature_name, dtype,
                               tensor_shape, indices_index, values_index);

  ATDSDecoder atds_decoder =
      ATDSDecoder(dense_features, sparse_features, varlen_features);
  Status init_status = atds_decoder.Initialize(writer_schema);
  ASSERT_TRUE(init_status.ok());

  std::vector<avro::GenericDatum> skipped_data = atds_decoder.GetSkippedData();
  std::vector<Tensor> dense_tensors;
  sparse::ValueBuffer buffer;
  sparse::GetValuesBuffer<string>(buffer).resize(1);
  buffer.indices.resize(1);
  buffer.num_of_elements.resize(1);
  Status decode_status =
      atds_decoder.DecodeATDSDatum(decoder, dense_tensors, buffer, skipped_data,
                                   static_cast<size_t>(offset));
  ASSERT_TRUE(decode_status.ok());

  std::vector<size_t> expected_num_elements = {expected_values.size()};

  ValidateBuffer(buffer, varlen_features[0], expected_indices, expected_values,
                 expected_num_elements);
}

TEST(VarlenDecoderTest, DT_INT32_scalar) {
  int value = -7;
  long offset = 1;
  std::vector<long> expected_indices = {offset};
  std::vector<int> expected_values = {value};

  VarlenDecoderTest(value, DT_INT32, {}, expected_indices, expected_values,
                    offset);
}

TEST(VarlenDecoderTest, DT_INT32_1D) {
  std::vector<int> values = {1, 2, 3};
  long offset = 9;
  std::vector<long> expected_indices = {offset, 0, offset, 1, offset, 2};
  std::vector<int> expected_values = values;

  VarlenDecoderTest(values, DT_INT32, {-1}, expected_indices, expected_values,
                    offset);
}

TEST(VarlenDecoderTest, DT_INT32_2D) {
  std::vector<std::vector<int>> values = {{-1}, {4, 5, 6}, {-7, 8}};
  long offset = 16;
  std::vector<long> expected_indices = {offset, 0, 0, offset, 1, 0,
                                        offset, 1, 1, offset, 1, 2,
                                        offset, 2, 0, offset, 2, 1};
  std::vector<int> expected_values = {-1, 4, 5, 6, -7, 8};

  VarlenDecoderTest(values, DT_INT32, {3, -1}, expected_indices,
                    expected_values, offset);
}

TEST(VarlenDecoderTest, DT_INT64_scalar) {
  long value = 1;
  long offset = 0;
  std::vector<long> expected_indices = {offset};
  std::vector<long> expected_values = {value};
  VarlenDecoderTest(value, DT_INT64, {}, expected_indices, expected_values,
                    offset);
}

TEST(VarlenDecoderTest, DT_INT64_1D) {
  std::vector<int64_t> values = {1};
  long offset = 3;
  std::vector<long> expected_indices = {offset, 0};
  std::vector<int64_t> expected_values = values;
  VarlenDecoderTest(values, DT_INT64, {-1}, expected_indices, expected_values,
                    offset);
}

TEST(VarlenDecoderTest, DT_INT64_2D) {
  std::vector<std::vector<int64_t>> values = {{1}};
  long offset = 3;
  std::vector<long> expected_indices = {offset, 0, 0};
  std::vector<int64_t> expected_values = {1};
  VarlenDecoderTest(values, DT_INT64, {-1, -1}, expected_indices,
                    expected_values, offset);
}

TEST(VarlenDecoderTest, DT_FLOAT_scalar) {
  float value = -0.6;
  long offset = 5;
  std::vector<long> expected_indices = {offset};
  std::vector<float> expected_values = {value};
  VarlenDecoderTest(value, DT_FLOAT, {}, expected_indices, expected_values,
                    offset);
}

TEST(VarlenDecoderTest, DT_FLOAT_1D) {
  std::vector<float> values = {};
  long offset = 111;
  std::vector<long> expected_indices = {};
  std::vector<float> expected_values = values;
  VarlenDecoderTest(values, DT_FLOAT, {-1}, expected_indices, expected_values,
                    offset);
}

TEST(VarlenDecoderTest, DT_FLOAT_2D) {
  std::vector<std::vector<float>> values = {{-0.1, -0.2, -0.3}, {-1.4, 5.4}};
  long offset = 111;
  std::vector<long> expected_indices = {
      offset, 0, 0, offset, 0, 1, offset, 0, 2, offset, 1, 0, offset, 1, 1};
  std::vector<float> expected_values = {-0.1, -0.2, -0.3, -1.4, 5.4};
  VarlenDecoderTest(values, DT_FLOAT, {-1, -1}, expected_indices,
                    expected_values, offset);
}

TEST(VarlenDecoderTest, DT_DOUBLE_scalar) {
  double value = -0.99;
  long offset = 1;
  std::vector<long> expected_indices = {offset};
  std::vector<double> expected_values = {value};
  VarlenDecoderTest(value, DT_DOUBLE, {}, expected_indices, expected_values,
                    offset);
}

TEST(VarlenDecoderTest, DT_DOUBLE_1D) {
  std::vector<double> values = {1.852, 0.79};
  long offset = 3;
  std::vector<long> expected_indices = {offset, 0, offset, 1};
  std::vector<double> expected_values = values;
  VarlenDecoderTest(values, DT_DOUBLE, {-1}, expected_indices, expected_values,
                    offset);
}

TEST(VarlenDecoderTest, DT_DOUBLE_2D) {
  std::vector<std::vector<double>> values = {};
  long offset = 5;
  std::vector<long> expected_indices = {};
  std::vector<double> expected_values = {};
  VarlenDecoderTest(values, DT_DOUBLE, {-1, 2}, expected_indices,
                    expected_values, offset);
}

TEST(VarlenDecoderTest, DT_STRING_scalar) {
  string value = "abc";
  long offset = 7;
  std::vector<long> expected_indices = {offset};
  std::vector<string> expected_values = {"abc"};
  VarlenDecoderTest(value, DT_STRING, {}, expected_indices, expected_values,
                    offset);
}

TEST(VarlenDecoderTest, DT_BYTES_scalar) {
  byte_array value{0xb4, 0x98, 0x1a};
  long offset = 7;
  std::vector<long> expected_indices = {offset};
  std::vector<byte_array> expected_values = {value};
  VarlenDecoderTest(value, DT_STRING, {}, expected_indices, expected_values,
                    offset, avro::AVRO_BYTES);
}

TEST(VarlenDecoderTest, DT_STRING_1D) {
  std::vector<string> values = {"", "", ""};
  long offset = 0;
  std::vector<long> expected_indices = {offset, 0, offset, 1, offset, 2};
  std::vector<string> expected_values = values;
  VarlenDecoderTest(values, DT_STRING, {-1}, expected_indices, expected_values,
                    offset);
}

TEST(VarlenDecoderTest, DT_BYTES_1D) {
  byte_array v1{0xb4, 0xaf, 0x98, 0x1a};
  byte_array v2{0xb4, 0xaf, 0x98};
  byte_array v3{0xb4, 0x98, 0x1a};
  std::vector<byte_array> values = {v1, v2, v3};
  long offset = 0;
  std::vector<long> expected_indices = {offset, 0, offset, 1, offset, 2};
  std::vector<byte_array> expected_values = values;
  VarlenDecoderTest(values, DT_STRING, {-1}, expected_indices, expected_values,
                    offset, avro::AVRO_BYTES);
}

TEST(VarlenDecoderTest, DT_STRING_2D) {
  std::vector<std::vector<string>> values = {{"abc"}, {"ABC"}, {"LINKEDIN"}};
  long offset = 0;
  std::vector<long> expected_indices = {offset, 0,      0, offset, 1,
                                        0,      offset, 2, 0};
  std::vector<string> expected_values = {"abc", "ABC", "LINKEDIN"};
  VarlenDecoderTest(values, DT_STRING, {-1, 1}, expected_indices,
                    expected_values, offset);
}

TEST(VarlenDecoderTest, DT_BYTES_2D) {
  byte_array v1{0xb4, 0xaf, 0x98, 0x1a};
  byte_array v2{0xb4, 0xaf, 0x98};
  byte_array v3{0xb4, 0x98, 0x1a};
  std::vector<std::vector<byte_array>> values = {{v1}, {v2}, {v3}};
  long offset = 0;
  std::vector<long> expected_indices = {offset, 0,      0, offset, 1,
                                        0,      offset, 2, 0};
  std::vector<byte_array> expected_values = {v1, v2, v3};
  VarlenDecoderTest(values, DT_STRING, {-1, 1}, expected_indices,
                    expected_values, offset, avro::AVRO_BYTES);
}

TEST(VarlenDecoderTest, DT_BOOL_scalar) {
  bool value = true;
  long offset = 0;
  std::vector<long> expected_indices = {offset};
  std::vector<bool> expected_values = {value};
  VarlenDecoderTest(value, DT_BOOL, {}, expected_indices, expected_values,
                    offset);
}

TEST(VarlenDecoderTest, DT_BOOL_1D) {
  std::vector<bool> values = {true, false, true};
  long offset = 3;
  std::vector<long> expected_indices = {offset, 0, offset, 1, offset, 2};
  std::vector<bool> expected_values = values;
  VarlenDecoderTest(values, DT_BOOL, {-1}, expected_indices, expected_values,
                    offset);
}

TEST(VarlenDecoderTest, DT_BOOL_2D) {
  std::vector<std::vector<bool>> values = {{}, {true, true}};
  long offset = 4;
  std::vector<long> expected_indices = {offset, 1, 0, offset, 1, 1};
  std::vector<bool> expected_values = {true, true};
  VarlenDecoderTest(values, DT_BOOL, {2, -1}, expected_indices, expected_values,
                    offset);
}

}  // namespace varlen
}  // namespace atds
}  // namespace tensorflow
