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
#ifndef TENSORFLOW_DATA_AVRO_PARSER_H_
#define TENSORFLOW_DATA_AVRO_PARSER_H_

#include <map>
#include <queue>
#include <set>
#include <vector>
#include "api/Generic.hh"
#include "api/Types.hh"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow_io/core/avro/utils/value_buffer.h"

namespace tensorflow {
namespace data {

// Avro parser
class AvroParser;
using AvroParserUniquePtr = std::unique_ptr<AvroParser>;
using AvroParserSharedPtr = std::shared_ptr<AvroParser>;

class AvroParser {
 public:
  // Constructor
  AvroParser(const string& key);

  // Virtual destructor ensures the derived class's destructor is called and
  // clean up its memory.
  virtual ~AvroParser() {}

  // This must be called before using parsers for arrays once the entire tree is
  // constructed
  void ComputeFinalDescendents();

  // Parse will traverse the sub-tree of this value and fill all values into
  // `parsed_values` may also read from parsed values if filtering
  virtual Status Parse(std::map<string, ValueStoreUniquePtr>* parsed_values,
                       const avro::GenericDatum& datum) const = 0;

  // Add a child to this avro parser
  inline void AddChild(const AvroParserSharedPtr& child) {
    children_.push_back(child);
  }

  // public for testing
  const std::vector<AvroParserSharedPtr> GetChildren() const;

  // Convert the avro parser into a human readable string representation
  virtual string ToString(size_t level = 0) const = 0;

  // Get the key for this avro parser -- this key can be used to map to values
  inline const string GetKey() const { return key_; }

  // Get the supported avro types for this parser
  virtual std::set<avro::Type> GetSupportedTypes() const = 0;

 protected:
  // Get the final descendents for this avro parser
  const std::vector<AvroParserSharedPtr> GetFinalDescendents() const;

  // Convert all children into a string representation
  string ChildrenToString(size_t level) const;

  // Convert the level into a string
  string LevelToString(size_t level) const;

  // Convert the supported types into a string representation
  string SupportedTypesToString(char separator) const;

  // The key for this avro parser
  string key_;

 private:
  // Is this a terminal parser node
  inline bool IsTerminal() const { return children_.size() == 0; }

  // Children for this avro parser
  std::vector<AvroParserSharedPtr> children_;

  // The final descendents of this parser are computed upon first call and then
  // cached
  std::vector<AvroParserSharedPtr> final_descendents_;
};

// Parser for primitive types
/*
// TODO(fraudies): Use this null value parser to resolve to default value
// if provided
class NullValueParser : public AvroParser {
public:
  NullValueParser(const string& key);
  // Don't do anything, meaning for dense tensors the default will be filled
  // and for sparse tensors no entry will be generated
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
    const avro_value_t& value) const override;
  virtual string ToString(int level = 0) const;
  inline avro_type_t GetSupportedTypes() const override { return AVRO_NULL; }
};
*/

// Parser for boolean values
class BoolValueParser : public AvroParser {
 public:
  BoolValueParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_BOOL};
  }
};

// Parser for long values
class LongValueParser : public AvroParser {
 public:
  LongValueParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_LONG};
  }
};

// Parser for integer values
class IntValueParser : public AvroParser {
 public:
  IntValueParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_INT};
  }
};

// Parser for double values
class DoubleValueParser : public AvroParser {
 public:
  DoubleValueParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_DOUBLE};
  }
};

// Parser for float values
class FloatValueParser : public AvroParser {
 public:
  FloatValueParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_FLOAT};
  }
};

// Parser for string or byte values
// Tries to parse a string first and if this fails will try to parse bytes
// second
class StringBytesEnumFixedValueParser : public AvroParser {
 public:
  StringBytesEnumFixedValueParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_STRING, avro::AVRO_BYTES, avro::AVRO_ENUM,
            avro::AVRO_FIXED};
  }
};

// Parser for an array -- parses all elements
class ArrayAllParser : public AvroParser {
 public:
  ArrayAllParser();
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_ARRAY};
  }
};

// Parser for an array -- parses one index out of the array
class ArrayIndexParser : public AvroParser {
 public:
  ArrayIndexParser(size_t index);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_ARRAY};
  }

 private:
  size_t index_;
};

// Parser for an array -- parses all values that evaluate the filter to true
class ArrayFilterParser : public AvroParser {
 public:
  enum ArrayFilterType { kLhsIsConstant, kRhsIsConstant, kNoConstant };
  ArrayFilterParser(const tstring& lhs, const tstring& rhs,
                    ArrayFilterType type);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  static ArrayFilterType ToArrayFilterType(bool lhs_is_constant,
                                           bool rhs_is_constant);
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_ARRAY};
  }

 private:
  tstring lhs_;
  tstring rhs_;
  ArrayFilterType type_;
};

// Parser for a map -- parses one key
class MapKeyParser : public AvroParser {
 public:
  MapKeyParser(const string& key);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_MAP};
  }

 private:
  string key_;  // key for map
};

// Parser for a record -- parses all attributes
class RecordParser : public AvroParser {
 public:
  RecordParser(const string& name);
  // check that the in_value is of type record
  // check that an attribute with name exists
  // get the the attribute for the name and return it in the vector as single
  // element
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_RECORD};
  }

 private:
  string name_;
};

// Parser for a union in a record
class UnionParser : public AvroParser {
 public:
  UnionParser(const string& type_name);
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_UNION};
  }

 private:
  string type_name_;
};

// Parses the namespace
class NamespaceParser : public AvroParser {
 public:
  NamespaceParser(const string& name);
  // checks namespace of value against given namespace
  // - if matches passes avro value to all it's child parsers
  // - if does not match returns error with actual and expected namespace
  Status Parse(std::map<string, ValueStoreUniquePtr>* values,
               const avro::GenericDatum& datum) const override;
  virtual string ToString(size_t level = 0) const;
  // Note, abuse of unknown symbol
  inline std::set<avro::Type> GetSupportedTypes() const override {
    return {avro::AVRO_UNKNOWN};
  }

 private:
  string name_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_AVRO_PARSER_H_
