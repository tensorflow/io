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

#include "tensorflow_io/core/kernels/avro/atds/sparse_feature_decoder.h"

#include "api/Decoder.hh"
#include "api/Stream.hh"
#include "api/ValidSchema.hh"
#include "tensorflow/core/platform/test.h"
#include "tensorflow_io/core/kernels/avro/atds/decoder_test_util.h"

namespace tensorflow {
namespace atds {
namespace sparse {

using Indices = std::vector<std::vector<long>>;

template <typename T>
void SparseDecoderTest(const Indices& indices, const std::vector<T>& values,
                       const std::vector<size_t>& order,
                       std::initializer_list<int64> shape, long offset,
                       const avro::Type avro_type = avro::AVRO_NULL) {
  DataType dtype = GetDataType<T>();
  string feature_name = "feature";
  ATDSSchemaBuilder schema_builder = ATDSSchemaBuilder();
  schema_builder.AddSparseFeature(feature_name, dtype, order, avro_type);

  string schema = schema_builder.Build();
  avro::ValidSchema writer_schema = schema_builder.BuildVaildSchema();
  avro::GenericDatum atds_datum(writer_schema);
  AddSparseValue(atds_datum, feature_name, indices, values);

  avro::OutputStreamPtr out_stream = EncodeAvroGenericDatum(atds_datum);
  avro::InputStreamPtr in_stream = avro::memoryInputStream(*out_stream);
  avro::DecoderPtr decoder = avro::binaryDecoder();
  decoder->init(*in_stream);

  std::vector<dense::Metadata> dense_features;
  std::vector<sparse::Metadata> sparse_features;
  std::vector<varlen::Metadata> varlen_features;
  size_t indices_index = 0, values_index = 0;
  PartialTensorShape tensor_shape(shape);
  sparse_features.emplace_back(FeatureType::sparse, feature_name, dtype,
                               tensor_shape, indices_index, values_index);

  ATDSDecoder atds_decoder =
      ATDSDecoder(dense_features, sparse_features, varlen_features);
  Status init_status = atds_decoder.Initialize(writer_schema);
  ASSERT_TRUE(init_status.ok());

  std::vector<avro::GenericDatum> skipped_data = atds_decoder.GetSkippedData();
  std::vector<Tensor> dense_tensors;
  ValueBuffer buffer;
  GetValuesBuffer<T>(buffer).resize(1);
  buffer.indices.resize(1);
  buffer.num_of_elements.resize(1);
  Status decode_status = atds_decoder.DecodeATDSDatum(
      decoder, dense_tensors, buffer, skipped_data, offset);
  ASSERT_TRUE(decode_status.ok());

  auto rank = indices.size();
  auto num_elem = values.size();
  std::vector<long> expected_indices((rank + 1) * num_elem, offset);
  for (size_t i = 0; i < indices.size(); i++) {
    auto dim = i + 1;
    for (size_t j = 0; j < indices[i].size(); j++) {
      expected_indices[dim + j * (rank + 1)] = indices[i][j];
    }
  }
  std::vector<size_t> expected_num_elements = {num_elem};

  ValidateBuffer(buffer, sparse_features[0], expected_indices, values,
                 expected_num_elements);
}

template <>
inline void SparseDecoderTest(const Indices& indices,
                              const std::vector<byte_array>& values,
                              const std::vector<size_t>& order,
                              std::initializer_list<int64> shape, long offset,
                              const avro::Type avro_type) {
  DataType dtype = DT_STRING;
  string feature_name = "feature";
  ATDSSchemaBuilder schema_builder = ATDSSchemaBuilder();
  schema_builder.AddSparseFeature(feature_name, dtype, order, avro_type);

  string schema = schema_builder.Build();
  avro::ValidSchema writer_schema = schema_builder.BuildVaildSchema();
  avro::GenericDatum atds_datum(writer_schema);
  AddSparseValue(atds_datum, feature_name, indices, values);

  avro::OutputStreamPtr out_stream = EncodeAvroGenericDatum(atds_datum);
  avro::InputStreamPtr in_stream = avro::memoryInputStream(*out_stream);
  avro::DecoderPtr decoder = avro::binaryDecoder();
  decoder->init(*in_stream);

  std::vector<dense::Metadata> dense_features;
  std::vector<sparse::Metadata> sparse_features;
  std::vector<varlen::Metadata> varlen_features;
  size_t indices_index = 0, values_index = 0;
  PartialTensorShape tensor_shape(shape);
  sparse_features.emplace_back(FeatureType::sparse, feature_name, dtype,
                               tensor_shape, indices_index, values_index);

  ATDSDecoder atds_decoder =
      ATDSDecoder(dense_features, sparse_features, varlen_features);
  Status init_status = atds_decoder.Initialize(writer_schema);
  ASSERT_TRUE(init_status.ok());

  std::vector<avro::GenericDatum> skipped_data = atds_decoder.GetSkippedData();
  std::vector<Tensor> dense_tensors;
  ValueBuffer buffer;
  GetValuesBuffer<string>(buffer).resize(1);
  buffer.indices.resize(1);
  buffer.num_of_elements.resize(1);
  Status decode_status = atds_decoder.DecodeATDSDatum(
      decoder, dense_tensors, buffer, skipped_data, offset);
  ASSERT_TRUE(decode_status.ok());

  auto rank = indices.size();
  auto num_elem = values.size();
  std::vector<long> expected_indices((rank + 1) * num_elem, offset);
  for (size_t i = 0; i < indices.size(); i++) {
    auto dim = i + 1;
    for (size_t j = 0; j < indices[i].size(); j++) {
      expected_indices[dim + j * (rank + 1)] = indices[i][j];
    }
  }
  std::vector<size_t> expected_num_elements = {num_elem};

  ValidateBuffer(buffer, sparse_features[0], expected_indices, values,
                 expected_num_elements);
}

TEST(SparseDecoderTest, DT_INT32_1D) {
  std::vector<int> values = {1, 2, 3};
  SparseDecoderTest({{1, 3, 5}}, values, {0, 1}, {10}, 0);
}

TEST(SparseDecoderTest, DT_INT32_2D) {
  std::vector<int> values = {-1, 2};
  SparseDecoderTest({{3, 5}, {2, 4}}, values, {0, 1, 2}, {10, 5}, 0);
}

TEST(SparseDecoderTest, DT_INT64_1D) {
  std::vector<int64_t> values = {4};
  SparseDecoderTest({{1}}, values, {0, 1}, {100}, 0);
}

TEST(SparseDecoderTest, DT_INT64_2D) {
  std::vector<int64_t> values = {77, 99, 131, 121};
  SparseDecoderTest({{3, 3, 3, 3}, {2, 4, 6, 8}}, values, {0, 1, 2}, {10, 9},
                    0);
}

TEST(SparseDecoderTest, DT_FLOAT_1D) {
  std::vector<float> values = {0.0};
  SparseDecoderTest({{0}}, values, {0, 1}, {10}, 0);
}

TEST(SparseDecoderTest, DT_FLOAT_2D) {
  std::vector<float> values = {1.0, 0.0};
  SparseDecoderTest({{3, 5}, {2, 4}}, values, {0, 1, 2}, {10, 5}, 0);
}

TEST(SparseDecoderTest, DT_DOUBLE_1D) {
  std::vector<double> values = {1.0, 2.0, 3.0};
  SparseDecoderTest({{1, 3, 5}}, values, {0, 1}, {256}, 0);
}

TEST(SparseDecoderTest, DT_DOUBLE_2D) {
  std::vector<double> values = {0.77, 0.3145};
  SparseDecoderTest({{0, 1}, {0, 1}}, values, {0, 1, 2}, {2, 2}, 0);
}

TEST(SparseDecoderTest, DT_STRING_1D) {
  std::vector<string> values = {"abc"};
  SparseDecoderTest({{1}}, values, {0, 1}, {100}, 0);
}

TEST(SparseDecoderTest, DT_STRING_2D) {
  std::vector<string> values = {"abc", "cdf", "pdf", "rdf"};
  SparseDecoderTest({{1000, 1200, 98742, 919101}, {10101, 9291, 0, 191}},
                    values, {0, 1, 2}, {1000000, 12000}, 0);
}

TEST(SparseDecoderTest, DT_BYTES_1D) {
  byte_array value = {0xb4, 0xaf, 0x98, 0x1a};
  std::vector<byte_array> values = {value};
  SparseDecoderTest({{1}}, values, {0, 1}, {100}, 0, avro::AVRO_BYTES);
}

TEST(SparseDecoderTest, DT_BYTES_2D) {
  byte_array v1{0xb4, 0xaf, 0x98, 0x1a};
  byte_array v2{0xb4, 0xaf, 0x98};
  byte_array v3{0xb4, 0x98, 0x1a};
  byte_array v4{0xb4, 0x98};
  std::vector<byte_array> values = {v1, v2, v3, v4};
  SparseDecoderTest({{1000, 1200, 98742, 919101}, {10101, 9291, 0, 191}},
                    values, {0, 1, 2}, {1000000, 12000}, 0, avro::AVRO_BYTES);
}

TEST(SparseDecoderTest, DT_BOOL_1D) {
  std::vector<bool> values = {true, false, true};
  SparseDecoderTest({{0, 1, 2}}, values, {0, 1}, {10}, 0);
}

TEST(SparseDecoderTest, DT_BOOL_2D) {
  std::vector<bool> values = {false, false, true};
  SparseDecoderTest({{3, 5, 5}, {2, 4, 8}}, values, {0, 1, 2}, {10, 10}, 0);
}

TEST(SparseDecoderTest, 2D_Order_0_2_1) {
  std::vector<int> values = {-1, 2};
  SparseDecoderTest({{3, 5}, {2, 4}}, values, {0, 2, 1}, {10, 5}, 0);
}

TEST(SparseDecoderTest, 2D_Order_2_0_1) {
  std::vector<int> values = {-1, 2};
  SparseDecoderTest({{3, 5}, {2, 4}}, values, {2, 0, 1}, {10, 5}, 0);
}

TEST(SparseDecoderTest, 2D_Order_2_1_0) {
  std::vector<int> values = {-1, 2};
  SparseDecoderTest({{3, 5}, {2, 4}}, values, {2, 1, 0}, {10, 5}, 0);
}

TEST(SparseDecoderTest, 2D_Order_1_2_0) {
  std::vector<int> values = {-1, 2};
  SparseDecoderTest({{3, 5}, {2, 4}}, values, {1, 2, 0}, {10, 5}, 0);
}

TEST(SparseDecoderTest, 2D_Order_1_0_2) {
  std::vector<int> values = {-1, 2};
  SparseDecoderTest({{3, 5}, {2, 4}}, values, {1, 0, 2}, {10, 5}, 0);
}

TEST(SparseDecoderTest, NonZeroOffset) {
  std::vector<int64_t> values = {77, 99, 131, 121};
  SparseDecoderTest({{3, 3, 3, 3}, {2, 4, 6, 8}}, values, {0, 1, 2}, {10, 9},
                    99);
}

}  // namespace sparse
}  // namespace atds
}  // namespace tensorflow
