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

#include <memory>
#include "gtest/gtest.h"
#include "tensorflow_io/avro/utils/avro_parser.h"

namespace tensorflow {
namespace data {

// ------------------------------------------------------------
// Tests for avro terminal types
// ------------------------------------------------------------
TEST(AvroParserTest, BoolValueParser) {

  const string key("dummyKey");
  std::vector<bool> field_values = {true, false};

  for (bool field_value : field_values) {

    BoolValueParser parser(key);
    std::map<string, ValueStoreUniquePtr> values;
    values.insert(std::make_pair(key, std::unique_ptr<BoolValueBuffer>(new BoolValueBuffer())));

    avro_value_t value;
    avro_generic_boolean_new(&value, field_value);
    EXPECT_EQ(parser.Parse(&values, value), Status::OK());
    EXPECT_EQ((*reinterpret_cast<BoolValueBuffer*>(values[key].get())).back(), field_value);
  }
}

TEST(AvroParserTest, IntValueParser) {

  const string key("dummyKey");
  std::vector<int> field_values = {std::numeric_limits<int>::min(), -1, 0, 1,
    std::numeric_limits<int>::max()};

  for (int field_value : field_values) {

    IntValueParser parser(key);
    std::map<string, ValueStoreUniquePtr> values;
    values.insert(std::make_pair(key, std::unique_ptr<IntValueBuffer>(new IntValueBuffer())));

    avro_value_t value;
    avro_generic_int_new(&value, field_value);
    EXPECT_EQ(parser.Parse(&values, value), Status::OK());
    EXPECT_EQ((*reinterpret_cast<IntValueBuffer*>(values[key].get())).back(), field_value);
  }
}

TEST(AvroParserTest, StringValueParser) {

  const string key("dummyKey");
  std::vector<string> field_values = {"", "a", "abc", "328983"};

  for (const string& field_value : field_values) {
    StringOrBytesValueParser parser(key);
    std::map<string, ValueStoreUniquePtr> values;
    values.insert(std::make_pair(key, std::unique_ptr<StringValueBuffer>(new StringValueBuffer())));

    avro_value_t value;
    avro_generic_string_new(&value, field_value.c_str());
    EXPECT_EQ(parser.Parse(&values, value), Status::OK());
    EXPECT_EQ((*reinterpret_cast<StringValueBuffer*>(values[key].get())).back(), field_value);
  }
}


// ------------------------------------------------------------
// Tests for avro intermediary types
// ------------------------------------------------------------
TEST(AttributeParserTest, Parse) {
  // Create the value and fill it with dummy data
  const string field_value = "Karl Gauss";
  const string schema =
          "{"
          "  \"type\": \"record\","
          "  \"name\": \"person\","
          "  \"fields\": ["
          "    { \"name\": \"name\", \"type\": \"string\" },"
          "    { \"name\": \"age\", \"type\": \"int\" }"
          "  ]"
          "}";

	avro_value_t value;
	avro_schema_t record_schema = nullptr;
	EXPECT_EQ(avro_schema_from_json_length(schema.data(), schema.length(), &record_schema), 0);

	avro_value_iface_t* record_class = avro_generic_class_from_schema(record_schema);
	EXPECT_EQ(avro_generic_value_new(record_class, &value), 0);

	size_t  field_count;
  EXPECT_EQ(avro_value_get_size(&value, &field_count), 0);
  EXPECT_EQ(field_count, 2);

  avro_value_t field;
  EXPECT_EQ(avro_value_get_by_index(&value, 0, &field, NULL), 0);
  EXPECT_EQ(avro_value_set_string(&field, field_value.data()), 0);

  EXPECT_EQ(avro_value_get_by_index(&value, 1, &field, NULL), 0);
  EXPECT_EQ(avro_value_set_int(&field, 139), 0);

  const string key = "person.name";
  RecordParser parser("name");
  parser.AddChild(std::unique_ptr<StringOrBytesValueParser>(new StringOrBytesValueParser(key)));

  std::map<string, ValueStoreUniquePtr> values; // empty on purpose
  values.insert(std::make_pair(key, std::unique_ptr<StringValueBuffer>(new StringValueBuffer())));

  EXPECT_EQ(parser.Parse(&values, value), Status::OK());
  EXPECT_EQ((*reinterpret_cast<StringValueBuffer*>(values[key].get())).back(), field_value);
}

TEST(ArrayAllParser, Parse) {
  // Create the value and fill it with dummy data
  const string key = "person.socials";

  const int n_num = 10;
  const string schema =
          "{"
          "  \"type\": \"record\","
          "  \"name\": \"person\","
          "  \"fields\": ["
          "    { \"name\": \"socials\", \"type\": {\"type\": \"array\", \"items\": \"int\"} }"
          "  ]"
          "}";

	avro_value_t value;
	avro_schema_t record_schema = nullptr;
	EXPECT_EQ(avro_schema_from_json_length(schema.data(), schema.length(), &record_schema), 0);

	avro_value_iface_t* record_class = avro_generic_class_from_schema(record_schema);
	EXPECT_EQ(avro_generic_value_new(record_class, &value), 0);

	size_t  field_count;
  EXPECT_EQ(avro_value_get_size(&value, &field_count), 0);
  EXPECT_EQ(field_count, 1);

  avro_value_t socials_field;
  avro_value_t element;
  size_t index;
  EXPECT_EQ(avro_value_get_by_index(&value, 0, &socials_field, NULL), 0);
  for (int i_num = 0; i_num < n_num; ++i_num) {
    EXPECT_EQ(avro_value_append(&socials_field, &element, &index), 0);
    EXPECT_EQ(i_num, index);
    EXPECT_EQ(avro_value_set_int(&element, i_num), 0);
  }

  // Define the parsers for the socials
  RecordParser socials_parser("socials");
  AvroParserSharedPtr parse_all_items = std::make_shared<ArrayAllParser>();
  AvroParserSharedPtr parse_ints = std::make_shared<IntValueParser>(key);
  socials_parser.AddChild(parse_all_items);
  (*parse_all_items).AddChild(parse_ints);

  // Define the values
  std::map<string, ValueStoreUniquePtr> values;
  values.insert(std::make_pair(key, std::unique_ptr<IntValueBuffer>(new IntValueBuffer())));

  EXPECT_EQ((*parse_all_items).Parse(&values, socials_field), Status::OK());

  // This is the buffer with the expected values
  IntValueBuffer expected_buffer;
  expected_buffer.BeginMark();
  for (int i_num = 0; i_num < n_num; ++i_num) {
    expected_buffer.Add(i_num);
  }
  expected_buffer.FinishMark();

  for (int i_num = 0; i_num < n_num; ++i_num) {
    EXPECT_TRUE((*reinterpret_cast<IntValueBuffer*>(values[key].get()))
      .ValuesMatchAtReverseIndex(expected_buffer, 1+i_num));
  }
}

struct Person {
  string name;
  int age;
  Person(const string& name, int age) : name(name), age(age) { }
};

TEST(ArrayFilterParser, Parse) {
  const std::vector<Person> persons = {Person("Carl", 33), Person("Mary", 22), Person("Carl", 12)};
  const string persons_name = "persons";
  const string age_name = "age";
  const string name_name = "name";
  const string persons_name_key = "persons.[*].name";
  const string persons_age_key = "persons.[name='Carl'].age";
  const string schema = "{"
                        "  \"type\":\"record\","
                        "  \"name\":\"contact\","
                        "  \"fields\":["
                        "      {"
                        "         \"name\":\"persons\","
                        "         \"type\":{"
                        "            \"type\":\"array\","
                        "            \"items\":{"
                        "               \"type\":\"record\","
                        "               \"name\":\"person\","
                        "               \"fields\":["
                        "                  {"
                        "                     \"name\":\"name\","
                        "                     \"type\":\"string\""
                        "                  },"
                        "                  {"
                        "                     \"name\":\"age\","
                        "                     \"type\":\"int\""
                        "                  }"
                        "               ]"
                        "            }"
                        "         }"
                        "      }"
                        "   ]"
                        "}";

	avro_value_t value;
	avro_schema_t record_schema = nullptr;
	EXPECT_EQ(avro_schema_from_json_length(schema.data(), schema.length(), &record_schema), 0);

	avro_value_iface_t* record_class = avro_generic_class_from_schema(record_schema);
	EXPECT_EQ(avro_generic_value_new(record_class, &value), 0);

	size_t  field_count;
  EXPECT_EQ(avro_value_get_size(&value, &field_count), 0);
  EXPECT_EQ(field_count, 1);

  avro_value_t persons_field, persons_element, person_name, person_age;
  size_t index;
  EXPECT_EQ(avro_value_get_by_name(&value, persons_name.c_str(), &persons_field, NULL), 0);
  for (const Person& person : persons) {
    EXPECT_EQ(avro_value_append(&persons_field, &persons_element, &index), 0);

    EXPECT_EQ(avro_value_get_by_name(&persons_element, name_name.c_str(), &person_name, NULL), 0);
    EXPECT_EQ(avro_value_set_string(&person_name, person.name.data()), 0);

    EXPECT_EQ(avro_value_get_by_name(&persons_element, age_name.c_str(), &person_age, NULL), 0);
    EXPECT_EQ(avro_value_set_int(&person_age, person.age), 0);
  }

  RecordParser persons_parser(persons_name);
  AvroParserSharedPtr parse_carls_ages = std::make_shared<ArrayFilterParser>(persons_name_key, "Carl",
    ArrayFilterParser::ArrayFilterType::kRhsIsConstant);
  AvroParserSharedPtr parse_ages = std::make_shared<RecordParser>(age_name);
  AvroParserSharedPtr parse_int = std::make_shared<IntValueParser>(persons_age_key);
  (*parse_ages).AddChild(parse_int);
  (*parse_carls_ages).AddChild(parse_ages);
  persons_parser.AddChild(parse_carls_ages);

  std::unique_ptr<StringValueBuffer> names(new StringValueBuffer());
  (*names).AddByRef("Carl"); (*names).AddByRef("Mary"); (*names).AddByRef("Carl");

  std::map<string, ValueStoreUniquePtr> values;
  values.insert(std::make_pair(persons_name_key, std::move(names)));
  values.insert(std::make_pair(persons_age_key, std::unique_ptr<IntValueBuffer>(new IntValueBuffer())));

  EXPECT_EQ(persons_parser.Parse(&values, value), Status::OK());

  IntValueBuffer expected_buffer;
  expected_buffer.BeginMark();
  const int carls_ages[] = {33, 12};
  for (int i_age = 0; i_age < 2; ++i_age) {
    expected_buffer.Add(carls_ages[i_age]);
  }
  expected_buffer.FinishMark();

  for (int i_age = 0; i_age < 2; ++i_age) {
    EXPECT_TRUE((*reinterpret_cast<IntValueBuffer*>(values[persons_age_key].get()))
      .ValuesMatchAtReverseIndex(expected_buffer, 1+i_age));
  }
}

}  // namespace data
}  // namespace tensorflow