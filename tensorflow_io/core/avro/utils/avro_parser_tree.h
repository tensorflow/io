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
#ifndef TENSORFLOW_DATA_AVRO_PARSER_TREE_H_
#define TENSORFLOW_DATA_AVRO_PARSER_TREE_H_

#include <vector>
#include "tensorflow_io/core/avro/utils/avro_parser.h"
#include "tensorflow_io/core/avro/utils/prefix_tree.h"

namespace tensorflow {
namespace data {

typedef std::pair<string, DataType> KeyWithType;

// This vector holds only unique elements
// If we try to append or prepend a duplicate item it is not added
template <typename T>
class VectorOfUniqueElements {
 public:
  // Prepend an element to the vector
  bool Prepend(const T& value) {
    auto info = unique.insert(value);
    if (info.second) {
      order.insert(order.begin(), value);
    }
    return info.second;
  }

  // Append an element to the vector
  bool Append(const T& value) {
    auto info = unique.insert(value);
    if (info.second) {
      order.push_back(value);
    }
    return info.second;
  }

  // Get an ordered representation as a vector
  // The order is defined by the use of the add/prepend methods
  const std::vector<T>& GetOrdered() const { return order; }

 private:
  std::set<T> unique;
  std::vector<T> order;
};

// Creates a parser tree for avro data
// Note, avro data can have nesting
class AvroParserTree {
 public:
  // Creates all parser nodes with the given namespace and for the given keys
  // with their types
  static Status Build(AvroParserTree* parser_tree, const string& avro_namespace,
                      const std::vector<KeyWithType>& keys_and_types);

  // Parses all values in a batch into the map keyed by the user-defined keys
  // that map to value stores
  Status ParseValues(std::map<string, ValueStoreUniquePtr>* key_to_value,
                     const std::function<bool(avro::GenericDatum&)> read_value,
                     const avro::ValidSchema& reader_schema,
                     uint64 values_to_parse, uint64* values_parsed) const;

  Status ParseValues(std::map<string, ValueStoreUniquePtr>* key_to_value,
                     const std::function<bool(avro::GenericDatum&)> read_value,
                     const avro::ValidSchema& reader_schema) const;

  // Returns the namespace for this parser
  inline string GetAvroNamespace() const { return avro_namespace_; }

  // Returns the root of the parser tree -- exposed for testing
  inline AvroParserSharedPtr getRoot() const { return root_; }

  // Returns a string representation of this parser tree
  // This string representation is human friendly
  // Do not use it to serialize the tree
  string ToString() const { return (*root_).ToString(); };

 private:
  // The separator that is expected in keys
  static constexpr const char kSeparator = '.';

  // The constant for all element keys
  static constexpr const char* const kArrayAllElements = "[*]";

  // The constant for the default namespace -- used when the user did not define
  // a namespace
  static constexpr const char* const kDefaultNamespace = "default";

  // Build the avro parser tree for the parent for the given children from a
  // prefix tree
  Status Build(AvroParser* parent,
               const std::vector<PrefixTreeNodeSharedPtr>& children);

  // Initialize will compute the final descendents for each node, called after
  // build
  void Initialize();

  // Resolve and set namespace
  // If no namespace has been provided aka avro_namespace = '', then this method
  // resolves to the default namespace
  string ResolveAndSetNamespace(const string& avro_namespace);

  // Creates the value parser for the given infix and user name
  // Note, that this method only creates value parsers for non-primitive avro
  // types, e.g. attributes, maps, arrays, arrays with filters
  Status CreateValueParser(AvroParserUniquePtr& value_parser,
                           const string& infix, const string& user_name) const;

  // Creates the value parser for primitive avro types for the given user name
  // and data type
  Status CreateFinalValueParser(AvroParserUniquePtr& value_parser,
                                const string& user_name,
                                DataType data_type) const;

  // Initializes value buffers for all keys
  Status InitializeValueBuffers(
      std::map<string, ValueStoreUniquePtr>* key_to_value) const;

  // Orders and resolves key types
  // The ordering is necessary to process any depends for filters first before
  // resolving the filtering expression
  // The resolving here refers to the resolving of filter expressions
  static std::vector<KeyWithType> OrderAndResolveKeyTypes(
      const std::vector<KeyWithType>& keys_and_types);

  // Validate that keys are unique
  static Status ValidateUniqueKeys(
      const std::vector<KeyWithType>& keys_and_types);

  // Add a begin mark to all value stores
  // This is used to mark the outer-most dimension as begun -- before any
  // element is added
  static Status AddBeginMarks(
      std::map<string, ValueStoreUniquePtr>* key_to_value);

  // Add a finish mark to all value stores
  // This is used to mark the outer-most dimension as finished -- before any
  // element is added
  static Status AddFinishMarks(
      std::map<string, ValueStoreUniquePtr>* key_to_value);

  // Resolve a filter name
  // Handles the inplace notation where we need to add all parent names
  // Handles the absolute reference notation with symbol @ where the symbol
  // needs to be removed
  static string ResolveFilterName(const string& user_name,
                                  const string& side_name,
                                  const string& filter_name);

  // Get all parts of a user name without accounting for the avro namespace
  static std::vector<string> GetPartsWithoutAvroNamespace(
      const string& user_name, const string& avro_namespace);

  // Removes the default avro namespace from the name -- if it is present
  static string RemoveDefaultAvroNamespace(const string& name);

  // Removes additional dots that would not be part of the user name
  // These are the dots in front of the symbols `[` and `:`
  static string RemoveAddedDots(const string& name);

  // Checks if the name contains a filter expression, e.g.
  // 'does.not.matter[abc=efg].more.does.not.matter'. If a filter is present
  // this method returns the left-hand-side name and the right-hand-side name
  static bool ContainsFilter(string* lhs_name, string* rhs_name,
                             const string& name);

  // Is the infix a filter expression? E.g. '[abc=efg]'?
  // If this is true returns the left-hand-side name and the right-hand-side
  // name
  static bool IsFilter(string* lhs, string* rhs, const string& infix);

  // Is the infix an array expression -- this is '[*]'?
  static bool IsArrayAll(const string& infix);

  // Is this infix an array index, e.g. '[5]'?
  // If this is true the corresponding index is returned
  static bool IsArrayIndex(int* index, const string& infix);

  // Is this infix is a map key, e.g. '['key']'?
  // If this is true the corresponding key for the map is returned
  static bool IsMapKey(string* key, const string& infix);

  // Is this infix is an attribute name? -- must be a valid avro name, regex
  // [A-Za-z_] See https://avro.apache.org/docs/1.8.2/spec.html#names
  static bool IsAttribute(const string& infix);

  // Is this infix a string constant? -- must be valid name, regex [S+]
  static bool IsStringConstant(string* constant, const string& infix);

  // Is this infix a branch? -- must be
  // [:boolean|:int|:long|:float|:double|:bytes|:string]
  static bool IsBranch(const string& infix);

  // Is the given namespace the default namespace
  static bool IsDefaultNamespace(const string& avro_namespace) {
    return avro_namespace == kDefaultNamespace;
  };

  // The avro namespace
  string avro_namespace_;

  // The parser root
  AvroParserSharedPtr root_;

  // used to preserve the order in the parse value method, InitValueBuffers
  // before each parse call
  std::vector<std::pair<string, DataType> > keys_and_types_;

  // This map is a helper for fast access of the data type that corresponds to
  // the key
  std::map<string, DataType> key_to_type_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_AVRO_PARSER_TREE_H_
