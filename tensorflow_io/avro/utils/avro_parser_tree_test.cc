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

#include "gtest/gtest.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow_io/avro/utils/avro_parser_tree.h"

// Note, these tests do not cover all avro types, because there are enough tests
// in avroc for that. Instead these tests only cover the wrapping in the mem readers
namespace tensorflow {
namespace data {

TEST(AvroParserTreeTest, BuildParserTree) {
  std::vector<std::pair<string, DataType> > keys_and_types = {
    std::make_pair("friends[2].name.first", DT_STRING),
    std::make_pair("friends[*].address[*].street", DT_STRING),
    std::make_pair("friends[*].job[*].coworker[*].name.first", DT_STRING),
    std::make_pair("car['nickname'].color", DT_STRING),
    std::make_pair("friends[gender='unknown'].name.first", DT_STRING),
    std::make_pair("friends[name.first=name.last].name.initial", DT_STRING)};
  AvroParserTree parser_tree;
  EXPECT_EQ(AvroParserTree::Build(&parser_tree, "default", keys_and_types), Status::OK());
  AvroParserSharedPtr root_parser = parser_tree.getRoot();
  NamespaceParser* namespace_parser = dynamic_cast<NamespaceParser*>(root_parser.get());
  EXPECT_TRUE(namespace_parser != nullptr);
  const std::vector<AvroParserSharedPtr>& children((*root_parser).GetChildren());
  EXPECT_EQ(children.size(), 2);
  const string actual(parser_tree.ToString());
  const string expected =
    "|---NamespaceParser(default)\n"
    "|   |---RecordParser(friends)\n"
    "|   |   |---ArrayAllParser\n"
    "|   |   |   |---RecordParser(name)\n"
    "|   |   |   |   |---RecordParser(last)\n"
    "|   |   |   |   |   |---StringOrBytesValue(friends[*].name.last)\n"
    "|   |   |   |   |---RecordParser(first)\n"
    "|   |   |   |   |   |---StringOrBytesValue(friends[*].name.first)\n"
    "|   |   |   |---RecordParser(gender)\n"
    "|   |   |   |   |---StringOrBytesValue(friends[*].gender)\n"
    "|   |   |   |---RecordParser(address)\n"
    "|   |   |   |   |---ArrayAllParser\n"
    "|   |   |   |   |   |---RecordParser(street)\n"
    "|   |   |   |   |   |   |---StringOrBytesValue(friends[*].address[*].street)\n"
    "|   |   |   |---RecordParser(job)\n"
    "|   |   |   |   |---ArrayAllParser\n"
    "|   |   |   |   |   |---RecordParser(coworker)\n"
    "|   |   |   |   |   |   |---ArrayAllParser\n"
    "|   |   |   |   |   |   |   |---RecordParser(name)\n"
    "|   |   |   |   |   |   |   |   |---RecordParser(first)\n"
    "|   |   |   |   |   |   |   |   |   |---StringOrBytesValue(friends[*].job[*].coworker[*].name.first)\n"
    "|   |   |---ArrayFilterParser(friends[*].gender=unknown)\n"
    "|   |   |   |---RecordParser(name)\n"
    "|   |   |   |   |---RecordParser(first)\n"
    "|   |   |   |   |   |---StringOrBytesValue(friends[gender='unknown'].name.first)\n"
    "|   |   |---ArrayFilterParser(friends[*].name.first=friends[*].name.last)\n"
    "|   |   |   |---RecordParser(name)\n"
    "|   |   |   |   |---RecordParser(initial)\n"
    "|   |   |   |   |   |---StringOrBytesValue(friends[name.first=name.last].name.initial)\n"
    "|   |   |---ArrayIndexParser(2)\n"
    "|   |   |   |---RecordParser(name)\n"
    "|   |   |   |   |---RecordParser(first)\n"
    "|   |   |   |   |   |---StringOrBytesValue(friends[2].name.first)\n"
    "|   |---RecordParser(car)\n"
    "|   |   |---MapKeyParser(nickname)\n"
    "|   |   |   |---RecordParser(color)\n"
    "|   |   |   |   |---StringOrBytesValue(car['nickname'].color)\n";

  EXPECT_EQ(actual, expected);
}

TEST(AvroParserTreeTest, ParseIntArray) {
  const string int_array_key = "int_array[*]";
  std::vector<std::pair<string, DataType> > keys_and_types = {
    std::make_pair(int_array_key, DT_INT32)
  };
  AvroParserTree parser_tree;
  EXPECT_EQ(AvroParserTree::Build(&parser_tree, "default", keys_and_types), Status::OK());

  const std::vector<int> int_values = {1, 2, 3, 4};
  const string schema = "{  \"type\":\"record\","
                        "   \"name\":\"values\","
                        "   \"fields\":["
                        "      {"
                        "         \"name\": \"int_array\","
                        "         \"type\":{"
                        "             \"type\": \"array\","
                        "             \"items\": \"int\""
                        "         }"
                        "      }"
                        "   ]"
                        "}";

  avro_value_t value;
  avro_schema_t record_schema = nullptr;
  EXPECT_EQ(avro_schema_from_json_length(schema.data(), schema.length(), &record_schema), 0);

  avro_value_iface_t* record_class = avro_generic_class_from_schema(record_schema);
  EXPECT_EQ(avro_generic_value_new(record_class, &value), 0);

  size_t field_count;
  EXPECT_EQ(avro_value_get_size(&value, &field_count), 0);
  EXPECT_EQ(field_count, 1);

  avro_value_t int_array_field;
  const string int_array_name = "int_array";
  EXPECT_EQ(avro_value_get_by_name(&value, int_array_name.c_str(), &int_array_field, NULL), 0);

  for (int i_value = 0; i_value < int_values.size(); ++i_value) {
    int int_value = int_values[i_value];
    avro_value_t int_field;
    size_t index;

    // Get the field, check index, and set int value
    EXPECT_EQ(avro_value_append(&int_array_field, &int_field, &index), 0);
    EXPECT_EQ(i_value, index);
    EXPECT_EQ(avro_value_set_int(&int_field, int_value), 0);
  }

  std::map<string, ValueStoreUniquePtr> key_to_value;
  std::vector<AvroValueSharedPtr> values;
  values.push_back(std::make_shared<avro_value_t>(value));
  EXPECT_EQ(parser_tree.ParseValues(&key_to_value, values), Status::OK());

  auto key_and_value = key_to_value.find(int_array_key);
  // Entry should exist
  EXPECT_FALSE(key_and_value == key_to_value.end());

  // Define shapes
  const TensorShape shape({1, 4});

  // Define expected values
  Tensor expected(DT_INT32, shape);
  auto expected_flat = expected.flat<int>();
  for (int i_value = 0; i_value < int_values.size(); ++i_value) {
    expected_flat(i_value) = int_values[i_value];
  }

  // Define defaults
  Tensor defaults(DT_INT32, shape);
  auto defaults_flat = defaults.flat<int>();
  for (int i_value = 0; i_value < int_values.size(); ++i_value) {
    defaults_flat(i_value) = 0;
  }

  // Allocate memory for actual
  Tensor actual(DT_INT32, shape);

  // Make dense tensor from buffer
  EXPECT_EQ((*(key_and_value->second)).MakeDense(&actual, shape, defaults), Status::OK());

  // actual and expected must match
  test::ExpectTensorEqual<int>(actual, expected);
}


TEST(AvroParserTreeTest, ParseIntValue) {
  const string int_value_name = "int_value";

  std::vector<std::pair<string, DataType> > keys_and_types = {
    std::make_pair(int_value_name, DT_INT32)
  };
  AvroParserTree parser_tree;
  EXPECT_EQ(AvroParserTree::Build(&parser_tree, "default", keys_and_types), Status::OK());

  const int int_value = 12;
  const string schema = "{  \"type\":\"record\","
                        "   \"name\":\"values\","
                        "   \"fields\":["
                        "      {"
                        "         \"name\":\"int_value\","
                        "         \"type\":\"int\""
                        "      }"
                        "   ]"
                        "}";

  avro_value_t value;
  avro_schema_t record_schema = nullptr;
  EXPECT_EQ(avro_schema_from_json_length(schema.data(), schema.length(), &record_schema), 0);

  avro_value_iface_t* record_class = avro_generic_class_from_schema(record_schema);
  EXPECT_EQ(avro_generic_value_new(record_class, &value), 0);

  size_t field_count;
  EXPECT_EQ(avro_value_get_size(&value, &field_count), 0);
  EXPECT_EQ(field_count, 1);

  avro_value_t int_field;
  EXPECT_EQ(avro_value_get_by_name(&value, int_value_name.c_str(), &int_field, NULL), 0);
  EXPECT_EQ(avro_value_set_int(&int_field, int_value), 0);

  std::map<string, ValueStoreUniquePtr> key_to_value;
  std::vector<AvroValueSharedPtr> values;
  values.push_back(std::make_shared<avro_value_t>(value));
  EXPECT_EQ(parser_tree.ParseValues(&key_to_value, values), Status::OK());

  auto key_and_value = key_to_value.find(int_value_name);
  // Entry should exist
  EXPECT_FALSE(key_and_value == key_to_value.end());

  // Define shapes
  const TensorShape shape({1});

  // Define expected values
  Tensor expected(DT_INT32, shape);
  auto expected_flat = expected.flat<int>();
  expected_flat(0) = int_value;

  // Define defaults
  Tensor defaults(DT_INT32, shape);
  auto defaults_flat = defaults.flat<int>();
  defaults_flat(0) = 0;

  // Allocate memory for actual
  Tensor actual(DT_INT32, shape);

  // Make dense tensor from buffer
  EXPECT_EQ((*(key_and_value->second)).MakeDense(&actual, shape, defaults), Status::OK());

  // actual and expected must match
  test::ExpectTensorEqual<int>(actual, expected);
}


}
}