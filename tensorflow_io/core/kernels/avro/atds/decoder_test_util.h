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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DECODER_TEST_UTIL_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DECODER_TEST_UTIL_H_

#include "api/Encoder.hh"
#include "api/GenericDatum.hh"
#include "api/Node.hh"
#include "api/Specific.hh"
#include "api/Stream.hh"
#include "api/ValidSchema.hh"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow_io/core/kernels/avro/atds/atds_decoder.h"

namespace tensorflow {
namespace atds {

using byte_array = std::vector<uint8_t>;

class ATDSSchemaBuilder {
 public:
  ATDSSchemaBuilder();

  ATDSSchemaBuilder& AddDenseFeature(
      const string& name, DataType dtype, size_t rank,
      const avro::Type avro_type = avro::AVRO_NULL);
  ATDSSchemaBuilder& AddSparseFeature(
      const string& name, DataType dtype, size_t rank,
      const avro::Type avro_type = avro::AVRO_NULL);
  ATDSSchemaBuilder& AddSparseFeature(
      const string& name, DataType dtype, const std::vector<size_t>& order,
      const avro::Type avro_type = avro::AVRO_NULL);
  ATDSSchemaBuilder& AddOpaqueContextualFeature(const string& name,
                                                const string& type);

  string Build();
  avro::ValidSchema BuildVaildSchema();

 private:
  void AddFeature(const string&);
  string BuildFeatureSchema(const string&, const string&);
  string BuildNullableFeatureSchema(const string&, const string&);
  string GenerateDataType(DataType, const avro::Type = avro::AVRO_NULL);
  string GenerateArrayType(DataType, size_t,
                           const avro::Type = avro::AVRO_NULL);

  string schema_;
  size_t num_of_features_;
};

template <typename T>
DataType GetDataType() {
  return DataTypeToEnum<T>().value;
}

template <>
inline DataType GetDataType<string>() {
  return DT_STRING;
}

inline std::vector<uint8_t> StringToByte(const std::string& s) {
  std::vector<uint8_t> result;
  result.reserve(s.size());
  std::copy(s.begin(), s.end(), std::back_inserter(result));
  return result;
}

inline std::string ByteToString(const std::vector<uint8_t>& t) {
  std::string result;
  std::copy(t.begin(), t.end(), std::back_inserter(result));
  return result;
}

// avro::Type is used to differentiate between byte and string, both of which
// map to datatype
template <typename T>
void AddDenseValue(avro::GenericDatum& datum, const string& name,
                   const T& value) {
  auto& record = datum.value<avro::GenericRecord>();
  auto& feature = record.field(name);
  feature.value<T>() = value;
}

template <typename T>
void AddDenseValue(avro::GenericDatum& datum, const string& name,
                   const std::vector<T>& values) {
  auto& record = datum.value<avro::GenericRecord>();
  auto& feature = record.field(name).value<avro::GenericArray>();
  auto& feature_values = feature.value();
  for (T value : values) {
    feature_values.emplace_back(value);
  }
}

template <>
inline void AddDenseValue(avro::GenericDatum& datum, const string& name,
                          const byte_array& value) {
  auto& record = datum.value<avro::GenericRecord>();
  auto& feature = record.field(name);
  feature.value<byte_array>() = value;
}

template <typename T>
inline void AddDenseValue(avro::GenericDatum& datum, const string& name,
                          const std::vector<std::vector<T>>& values) {
  auto& record = datum.value<avro::GenericRecord>();
  auto& feature = record.field(name).value<avro::GenericArray>();
  auto& sub_array_schema = feature.schema()->leafAt(0);

  auto& feature_values = feature.value();
  for (size_t i = 0; i < values.size(); i++) {
    feature_values.emplace_back(sub_array_schema);
    auto& sub_array = feature_values.back().value<avro::GenericArray>().value();
    for (size_t j = 0; j < values[i].size(); j++) {
      sub_array.emplace_back(values[i][j]);
    }
  }
}

template <>
inline void AddDenseValue(avro::GenericDatum& datum, const string& name,
                          const std::vector<byte_array>& values) {
  auto& record = datum.value<avro::GenericRecord>();
  auto& feature = record.field(name).value<avro::GenericArray>();
  auto& feature_values = feature.value();
  for (byte_array value : values) {
    feature_values.emplace_back(value);
  }
}

template <typename T>
void AddSparseValue(avro::GenericDatum& datum, const string& name,
                    const std::vector<std::vector<long>>& indices,
                    const std::vector<T>& values) {
  auto& record = datum.value<avro::GenericRecord>();
  auto& feature = record.field(name).value<avro::GenericRecord>();

  for (size_t i = 0; i < indices.size(); i++) {
    auto indices_key = "indices" + std::to_string(i);
    auto& indices_array =
        feature.field(indices_key).value<avro::GenericArray>().value();
    for (long index : indices[i]) {
      indices_array.emplace_back(static_cast<int64_t>(index));
    }
  }

  auto& values_array =
      feature.field("values").value<avro::GenericArray>().value();
  for (T value : values) {
    values_array.emplace_back(value);
  }
}

avro::OutputStreamPtr EncodeAvroGenericDatum(avro::GenericDatum& datum);
avro::OutputStreamPtr EncodeAvroGenericData(
    std::vector<avro::GenericDatum>& data);

template <typename T, typename F>
void AssertValueEqual(const T& v1, const F& v2) {
  ASSERT_EQ(v1, v2);
}

template <>
inline void AssertValueEqual(const avro::NodePtr& v1, const avro::NodePtr& v2) {
  ASSERT_EQ(v1->type(), v2->type());
  ASSERT_EQ(v1->leaves(), v2->leaves());
  for (size_t i = 0; i < v1->leaves(); i++) {
    AssertValueEqual(v1->leafAt(i), v2->leafAt(i));
  }
}

template <>
inline void AssertValueEqual(const avro::ValidSchema& v1,
                             const avro::ValidSchema& v2) {
  AssertValueEqual(v1.root(), v2.root());
}

template <>
inline void AssertValueEqual(const tstring& v1, const string& v2) {
  ASSERT_STREQ(v1.c_str(), v2.c_str());
}

template <>
inline void AssertValueEqual(const string& v1, const tstring& v2) {
  ASSERT_STREQ(v1.c_str(), v2.c_str());
}

inline void AssertValueEqual(const char* v1, const char* v2, int len) {
  for (int i = 0; i < len; i++) {
    ASSERT_EQ(v1[i], v2[i]);
  }
}

template <>
inline void AssertValueEqual(const float& v1, const float& v2) {
  ASSERT_NEAR(v1, v2, 1e-6);
}

template <>
inline void AssertValueEqual(const double& v1, const double& v2) {
  ASSERT_NEAR(v1, v2, 1e-6);
}

template <typename T, typename U>
void AssertVectorValues(const std::vector<T>& actual,
                        const std::vector<U>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); i++) {
    AssertValueEqual(actual[i], expected[i]);
  }
}

template <typename T>
inline void AssertVectorValues(const std::vector<T>& actual,
                               const std::vector<byte_array>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); i++) {
    AssertValueEqual(actual[i], ByteToString(expected[i]));
  }
}

template <typename T>
void AssertTensorValues(const Tensor& tensor, const T& scalar) {
  AssertValueEqual(tensor.scalar<T>()(), scalar);
}

template <>
inline void AssertTensorValues(const Tensor& tensor, const string& scalar) {
  AssertValueEqual(tensor.scalar<tstring>()(), scalar);
}

template <typename T>
void AssertTensorValues(const Tensor& tensor, const std::vector<T>& vec) {
  for (size_t i = 0; i < vec.size(); i++) {
    AssertValueEqual(tensor.vec<T>()(i), vec[i]);
  }
  ASSERT_EQ(tensor.NumElements(), vec.size());
}

template <>
inline void AssertTensorValues(const Tensor& tensor, const byte_array& scalar) {
  AssertValueEqual(tensor.scalar<tstring>()(), ByteToString(scalar));
}

template <>
inline void AssertTensorValues(const Tensor& tensor,
                               const std::vector<string>& vec) {
  for (size_t i = 0; i < vec.size(); i++) {
    AssertValueEqual(tensor.vec<tstring>()(i), vec[i]);
  }
  ASSERT_EQ(tensor.NumElements(), vec.size());
}

template <typename T>
void AssertTensorValues(const Tensor& tensor,
                        const std::vector<std::vector<T>>& matrix) {
  size_t size = 0;
  for (size_t i = 0; i < matrix.size(); i++) {
    for (size_t j = 0; j < matrix[i].size(); j++) {
      AssertValueEqual(tensor.matrix<T>()(i, j), matrix[i][j]);
    }
    size += matrix[i].size();
  }
  ASSERT_EQ(tensor.NumElements(), size);
}

template <>
inline void AssertTensorValues(const Tensor& tensor,
                               const std::vector<byte_array>& vec) {
  for (size_t i = 0; i < vec.size(); i++) {
    AssertValueEqual(tensor.vec<tstring>()(i), ByteToString(vec[i]));
  }
  ASSERT_EQ(tensor.NumElements(), vec.size());
}

template <>
inline void AssertTensorValues(const Tensor& tensor,
                               const std::vector<std::vector<string>>& matrix) {
  size_t size = 0;
  for (size_t i = 0; i < matrix.size(); i++) {
    for (size_t j = 0; j < matrix[i].size(); j++) {
      AssertValueEqual(tensor.matrix<tstring>()(i, j), matrix[i][j]);
    }
    size += matrix[i].size();
  }
  ASSERT_EQ(tensor.NumElements(), size);
}

template <>
inline void AssertTensorValues(
    const Tensor& tensor, const std::vector<std::vector<byte_array>>& matrix) {
  size_t size = 0;
  for (size_t i = 0; i < matrix.size(); i++) {
    for (size_t j = 0; j < matrix[i].size(); j++) {
      AssertValueEqual(tensor.matrix<tstring>()(i, j),
                       ByteToString(matrix[i][j]));
    }
    size += matrix[i].size();
  }
  ASSERT_EQ(tensor.NumElements(), size);
}

template <typename T>
void AssertTensorRangeEqual(const Tensor& tensor, std::vector<T> values,
                            size_t offset) {
  for (size_t i = 0; i < values.size(); i++) {
    T actual = tensor.vec<T>()(offset + i);
    AssertValueEqual(actual, values[i]);
  }
}

template <>
inline void AssertTensorRangeEqual(const Tensor& tensor,
                                   std::vector<string> values, size_t offset) {
  for (size_t i = 0; i < values.size(); i++) {
    tstring actual = tensor.vec<tstring>()(offset + i);
    AssertValueEqual(actual, values[i]);
  }
}

template <typename T, typename Metadata>
void ValidateBuffer(sparse::ValueBuffer& buffer, const Metadata& metadata,
                    std::vector<long> indices, std::vector<T> values,
                    std::vector<size_t> num_of_elements) {
  size_t indices_index = metadata.indices_index;
  size_t values_index = metadata.values_index;

  AssertVectorValues(buffer.indices[indices_index], indices);
  std::vector<T>& actual_values =
      sparse::GetValueVector<T>(buffer, values_index);
  AssertVectorValues(actual_values, values);
  AssertVectorValues(buffer.num_of_elements[indices_index], num_of_elements);
}

template <typename Metadata>
void ValidateBuffer(sparse::ValueBuffer& buffer, const Metadata& metadata,
                    std::vector<long> indices, std::vector<byte_array> values,
                    std::vector<size_t> num_of_elements) {
  size_t indices_index = metadata.indices_index;
  size_t values_index = metadata.values_index;

  AssertVectorValues(buffer.indices[indices_index], indices);
  std::vector<string>& actual_values =
      sparse::GetValueVector<string>(buffer, values_index);
  AssertVectorValues(actual_values, values);
  AssertVectorValues(buffer.num_of_elements[indices_index], num_of_elements);
}

namespace sparse {

template <typename T>
std::vector<std::vector<T>>& GetValuesBuffer(ValueBuffer& buffer);

template <>
inline std::vector<std::vector<int>>& GetValuesBuffer(ValueBuffer& buffer) {
  return buffer.int_values;
}

template <>
inline std::vector<std::vector<long>>& GetValuesBuffer(ValueBuffer& buffer) {
  return buffer.long_values;
}

template <>
inline std::vector<std::vector<float>>& GetValuesBuffer(ValueBuffer& buffer) {
  return buffer.float_values;
}

template <>
inline std::vector<std::vector<double>>& GetValuesBuffer(ValueBuffer& buffer) {
  return buffer.double_values;
}

template <>
inline std::vector<std::vector<string>>& GetValuesBuffer(ValueBuffer& buffer) {
  return buffer.string_values;
}

template <>
inline std::vector<std::vector<bool>>& GetValuesBuffer(ValueBuffer& buffer) {
  return buffer.bool_values;
}

}  // namespace sparse

}  // namespace atds
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_DECODER_TEST_UTIL_H_
