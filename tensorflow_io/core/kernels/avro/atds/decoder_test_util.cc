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

#include "tensorflow_io/core/kernels/avro/atds/decoder_test_util.h"

#include "api/Compiler.hh"
#include "api/Generic.hh"
#include "api/Specific.hh"
#include "api/ValidSchema.hh"

namespace tensorflow {
namespace atds {

constexpr const char kATDSSchemaPrefix[] =
    "{"
    "\"type\" : \"record\", "
    "\"name\" : \"AvroTensorDataset\", "
    "\"namespace\" : \"com.organization.avrotensordataset\", "
    "\"fields\" : [ ";

constexpr const char kATDSSchemaSuffix[] =
    " ] "
    "}";

ATDSSchemaBuilder::ATDSSchemaBuilder()
    : schema_(kATDSSchemaPrefix), num_of_features_(0) {}

ATDSSchemaBuilder& ATDSSchemaBuilder::AddDenseFeature(
    const string& name, DataType dtype, size_t rank,
    const avro::Type avro_type) {
  string type = GenerateArrayType(dtype, rank, avro_type);
  string feature_schema = BuildFeatureSchema(name, type);
  AddFeature(feature_schema);
  return *this;
}

ATDSSchemaBuilder& ATDSSchemaBuilder::AddSparseFeature(
    const string& name, DataType dtype, size_t rank,
    const avro::Type avro_type) {
  std::vector<size_t> order(rank + 1, 0);
  for (size_t i = 0; i < order.size(); i++) {
    order[i] = i;
  }
  AddSparseFeature(name, dtype, order, avro_type);
  return *this;
}

ATDSSchemaBuilder& ATDSSchemaBuilder::AddSparseFeature(
    const string& name, DataType dtype, const std::vector<size_t>& order,
    const avro::Type avro_type) {
  string indices_type = GenerateArrayType(DT_INT64, 1);
  string values_type = GenerateArrayType(dtype, 1, avro_type);
  string fields = "";

  auto values_index = order.size() - 1;
  for (size_t i = 0; i < order.size(); i++) {
    if (i > 0) {
      fields += ", ";
    }
    if (order[i] == values_index) {
      fields += BuildFeatureSchema("values", values_type);
    } else {
      auto indices_name = "indices" + std::to_string(order[i]);
      fields += BuildFeatureSchema(indices_name, indices_type);
    }
  }

  string type =
      "{"
      "\"type\" : \"record\", "
      "\"name\" : \"" +
      name +
      "\", "
      "\"fields\" : [ " +
      fields +
      " ] "
      "}";
  string feature_schema = BuildFeatureSchema(name, type);
  AddFeature(feature_schema);
  return *this;
}

ATDSSchemaBuilder& ATDSSchemaBuilder::AddOpaqueContextualFeature(
    const string& name, const string& type) {
  string feature_schema = BuildFeatureSchema(name, type);
  AddFeature(feature_schema);
  return *this;
}

string ATDSSchemaBuilder::Build() { return schema_ + kATDSSchemaSuffix; }

avro::ValidSchema ATDSSchemaBuilder::BuildVaildSchema() {
  string schema = Build();

  std::istringstream iss(schema);
  avro::ValidSchema valid_schema;
  avro::compileJsonSchema(iss, valid_schema);
  return valid_schema;
}

void ATDSSchemaBuilder::AddFeature(const string& feature_schema) {
  if (num_of_features_ > 0) {
    schema_ += ", ";
  }
  schema_ += feature_schema;
  num_of_features_++;
}

string ATDSSchemaBuilder::BuildFeatureSchema(const string& name,
                                             const string& type) {
  return "{"
         "\"name\" : \"" +
         name +
         "\", "
         "\"type\" : " +
         type + " }";
}

string ATDSSchemaBuilder::BuildNullableFeatureSchema(const string& name,
                                                     const string& type) {
  return "{"
         "\"name\" : \"" +
         name +
         "\", "
         "\"type\" : [ \"null\", " +
         type +
         " ] "
         "}";
}

string ATDSSchemaBuilder::GenerateDataType(DataType dtype,
                                           const avro::Type avro_type) {
  switch (dtype) {
    case DT_INT32: {
      return "\"int\"";
    }
    case DT_INT64: {
      return "\"long\"";
    }
    case DT_FLOAT: {
      return "\"float\"";
    }
    case DT_DOUBLE: {
      return "\"double\"";
    }
    case DT_STRING: {
      if (avro_type == avro::AVRO_BYTES) {
        return "\"bytes\"";
      }
      return "\"string\"";
    }
    case DT_BOOL: {
      return "\"boolean\"";
    }
    default: {
      return "";
    }
  }
}

string ATDSSchemaBuilder::GenerateArrayType(DataType dtype, size_t rank,
                                            const avro::Type avro_type) {
  if (rank == 0) {
    return GenerateDataType(dtype, avro_type);
  }

  string type = GenerateArrayType(dtype, rank - 1, avro_type);
  return "{"
         "\"type\" : \"array\", "
         "\"items\" : " +
         type + " }";
}

avro::OutputStreamPtr EncodeAvroGenericDatum(avro::GenericDatum& datum) {
  avro::EncoderPtr encoder = avro::binaryEncoder();
  avro::OutputStreamPtr out_stream = avro::memoryOutputStream();
  encoder->init(*out_stream);
  avro::encode(*encoder, datum);
  encoder->flush();
  return std::move(out_stream);
}

avro::OutputStreamPtr EncodeAvroGenericData(
    std::vector<avro::GenericDatum>& data) {
  avro::EncoderPtr encoder = avro::binaryEncoder();
  avro::OutputStreamPtr out_stream = avro::memoryOutputStream();
  encoder->init(*out_stream);
  for (auto& datum : data) {
    avro::encode(*encoder, datum);
  }
  encoder->flush();
  return std::move(out_stream);
}

}  // namespace atds
}  // namespace tensorflow
