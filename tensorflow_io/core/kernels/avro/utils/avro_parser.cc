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
#include "tensorflow_io/core/kernels/avro/utils/avro_parser.h"

#include <queue>
#include <sstream>

namespace tensorflow {
namespace data {

// ------------------------------------------------------------
// AvroParser
// ------------------------------------------------------------
AvroParser::AvroParser(const string& key) : key_(key) {}

const std::vector<AvroParserSharedPtr> AvroParser::GetChildren() const {
  return children_;
}

void AvroParser::ComputeFinalDescendents() {
  std::queue<AvroParserSharedPtr> current;
  const std::vector<AvroParserSharedPtr>& children = GetChildren();
  for (const auto& child : children) {
    current.push(child);
  }
  // Helper variable for children of subsequent nodes
  while (!current.empty()) {
    if ((*current.front()).IsTerminal()) {
      final_descendents_.push_back(current.front());
    } else {
      const std::vector<AvroParserSharedPtr>& children =
          (*current.front()).GetChildren();
      for (const auto& child : children) {
        current.push(child);
      }
    }
    current.pop();
  }
}

const std::vector<AvroParserSharedPtr> AvroParser::GetFinalDescendents() const {
  // Return the final descendents
  return final_descendents_;
}

string AvroParser::ChildrenToString(size_t level) const {
  std::stringstream ss;
  for (const auto child : children_) {
    ss << (*child).ToString(level + 1);
  }
  return ss.str();
}

string AvroParser::LevelToString(size_t level) const {
  std::stringstream ss;
  for (size_t l = 0; l < level; ++l) {
    ss << "|   ";
  }
  return ss.str();
}

// Null will only resolve if
// -- the null appears in a union with a primitive type -- by construction
// (otherwise no default)
// -- the user supplied a scalar default tensor
// -- the user supplied a type that matches the primitive type
// For sparse tensors will null entries in their values or indices
// this will fail which is the correct behavior.
Status CheckValidDefault(const string& key,
                         const std::map<string, Tensor>& defaults,
                         DataType expected) {
  if (defaults.find(key) == defaults.end()) {
    return errors::InvalidArgument("For key '", key,
                                   "' cannot find a default value.");
  }
  const Tensor& default_value = defaults.at(key);
  if (!TensorShapeUtils::IsScalar(default_value.shape())) {
    return errors::InvalidArgument(
        "For key '", key,
        "' expected scalar default but got tensor with shape ",
        default_value.shape());
  }
  if (expected != default_value.dtype()) {
    return errors::InvalidArgument("For key '", key, "' expected data type ",
                                   expected, "' but got data type '",
                                   default_value.dtype(), "'.");
  }
  return OkStatus();
}

// This implementation assumes there is at least one expected type
string TypeErrorMessage(const std::set<avro::Type>& expected,
                        avro::Type actual) {
  string message = "";
  for (avro::Type t : expected) {
    message += ", '" + toString(t) + "'";
  }
  // Remove 2 for leading comma and blank -- here we assume there is at least
  // one
  return "Expected types: " + message.substr(2) + " but got type " +
         toString(actual) + ".";
}

BoolValueParser::BoolValueParser(const string& key) : AvroParser(key) {}
Status BoolValueParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
                              const avro::GenericDatum& datum,
                              const std::map<string, Tensor>& defaults) const {
  bool value;
  if (datum.type() == avro::AVRO_BOOL) {
    value = datum.value<bool>();
  } else if (datum.type() == avro::AVRO_NULL) {
    TF_RETURN_IF_ERROR(CheckValidDefault(key_, defaults, DT_BOOL));
    value = defaults.at(key_).flat<bool>()(0);
  } else {
    return errors::InvalidArgument(
        TypeErrorMessage(GetSupportedTypes(), datum.type()));
  }

  (*reinterpret_cast<BoolValueBuffer*>((*values).at(key_).get())).Add(value);
  return OkStatus();
}
string BoolValueParser::ToString(size_t level) const {
  return LevelToString(level) + "|---BoolValue(" + key_ + ")\n";
}

LongValueParser::LongValueParser(const string& key) : AvroParser(key) {}
Status LongValueParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
                              const avro::GenericDatum& datum,
                              const std::map<string, Tensor>& defaults) const {
  long value;
  if (datum.type() == avro::AVRO_LONG) {
    value = datum.value<long>();
  } else if (datum.type() == avro::AVRO_NULL) {
    TF_RETURN_IF_ERROR(CheckValidDefault(key_, defaults, DT_INT64));
    value = defaults.at(key_).flat<int64>()(0);
  } else {
    return errors::InvalidArgument(
        TypeErrorMessage(GetSupportedTypes(), datum.type()));
  }

  // Assume the key exists and cast is possible
  (*reinterpret_cast<LongValueBuffer*>((*values).at(key_).get())).Add(value);

  return OkStatus();
}

string LongValueParser::ToString(size_t level) const {
  return LevelToString(level) + "|---LongValue(" + key_ + ")\n";
}

IntValueParser::IntValueParser(const string& key) : AvroParser(key) {}
Status IntValueParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
                             const avro::GenericDatum& datum,
                             const std::map<string, Tensor>& defaults) const {
  int value;
  if (datum.type() == avro::AVRO_INT) {
    value = datum.value<int>();
  } else if (datum.type() == avro::AVRO_NULL) {
    TF_RETURN_IF_ERROR(CheckValidDefault(key_, defaults, DT_INT32));
    value = defaults.at(key_).flat<int32>()(0);
  } else {
    return errors::InvalidArgument(
        TypeErrorMessage(GetSupportedTypes(), datum.type()));
  }
  // Assume the key exists and cast is possible
  (*reinterpret_cast<IntValueBuffer*>((*values).at(key_).get())).Add(value);

  return OkStatus();
}

string IntValueParser::ToString(size_t level) const {
  return LevelToString(level) + "|---IntValue(" + key_ + ")\n";
}

DoubleValueParser::DoubleValueParser(const string& key) : AvroParser(key) {}
Status DoubleValueParser::Parse(
    std::map<string, ValueStoreUniquePtr>* values,
    const avro::GenericDatum& datum,
    const std::map<string, Tensor>& defaults) const {
  double value;
  if (datum.type() == avro::AVRO_DOUBLE) {
    value = datum.value<double>();
  } else if (datum.type() == avro::AVRO_NULL) {
    TF_RETURN_IF_ERROR(CheckValidDefault(key_, defaults, DT_DOUBLE));
    value = defaults.at(key_).flat<double>()(0);
  } else {
    return errors::InvalidArgument(
        TypeErrorMessage(GetSupportedTypes(), datum.type()));
  }
  // Assume the key exists and cast is possible
  (*reinterpret_cast<DoubleValueBuffer*>((*values)[key_].get())).Add(value);

  return OkStatus();
}

string DoubleValueParser::ToString(size_t level) const {
  return LevelToString(level) + "|---DoubleValue(" + key_ + ")\n";
}

FloatValueParser::FloatValueParser(const string& key) : AvroParser(key) {}
Status FloatValueParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
                               const avro::GenericDatum& datum,
                               const std::map<string, Tensor>& defaults) const {
  float value;
  if (datum.type() == avro::AVRO_FLOAT) {
    value = datum.value<float>();
  } else if (datum.type() == avro::AVRO_NULL) {
    TF_RETURN_IF_ERROR(CheckValidDefault(key_, defaults, DT_FLOAT));
    value = defaults.at(key_).flat<float>()(0);
  } else {
    return errors::InvalidArgument(
        TypeErrorMessage(GetSupportedTypes(), datum.type()));
  }
  // Assume the key exists and cast is possible
  (*reinterpret_cast<FloatValueBuffer*>((*values)[key_].get())).Add(value);

  return OkStatus();
}

string FloatValueParser::ToString(size_t level) const {
  return LevelToString(level) + "|---FloatValue(" + key_ + ")\n";
}

StringBytesEnumFixedValueParser::StringBytesEnumFixedValueParser(
    const string& key)
    : AvroParser(key) {}
Status StringBytesEnumFixedValueParser::Parse(
    std::map<string, ValueStoreUniquePtr>* values,
    const avro::GenericDatum& datum,
    const std::map<string, Tensor>& defaults) const {
  string value;
  switch (datum.type()) {
    case avro::AVRO_STRING:
      value = datum.value<string>();
      break;
    case avro::AVRO_BYTES: {
      const std::vector<uint8_t>& v = datum.value<std::vector<uint8_t>>();
      if (v.size() > 0) {
        value.resize(v.size());
        memcpy(&value[0], &v[0], v.size());
      }
    } break;
    case avro::AVRO_ENUM:
      value = datum.value<avro::GenericEnum>().symbol();
      break;
    case avro::AVRO_FIXED: {
      const std::vector<uint8_t>& v = datum.value<avro::GenericFixed>().value();
      if (v.size() > 0) {
        value.resize(v.size());
        memcpy(&value[0], &v[0], v.size());
      }
    } break;
    case avro::AVRO_NULL:
      TF_RETURN_IF_ERROR(CheckValidDefault(key_, defaults, DT_STRING));
      value = defaults.at(key_).flat<tstring>()(0);
      break;
    default:
      return errors::InvalidArgument(
          TypeErrorMessage(GetSupportedTypes(), datum.type()));
  }
  // Assume the key exists and cast is possible
  (*reinterpret_cast<StringValueBuffer*>((*values)[key_].get()))
      .AddByRef(value);

  return OkStatus();
}
string StringBytesEnumFixedValueParser::ToString(size_t level) const {
  return LevelToString(level) + "|---StringBytesEnumFixedValue(" + key_ + ")\n";
}

// ------------------------------------------------------------
// Concrete implementations of value parsers
// ------------------------------------------------------------
ArrayAllParser::ArrayAllParser() : AvroParser("") {}
Status ArrayAllParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
                             const avro::GenericDatum& datum,
                             const std::map<string, Tensor>& defaults) const {
  if (datum.type() != avro::AVRO_ARRAY) {
    return errors::InvalidArgument(
        TypeErrorMessage(GetSupportedTypes(), datum.type()));
  }

  std::vector<avro::GenericDatum> data =
      datum.value<avro::GenericArray>().value();

  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  const std::vector<AvroParserSharedPtr>& final_descendents(
      GetFinalDescendents());

  // Add a begin mark to all value buffers under this array
  for (const AvroParserSharedPtr& value_parser : final_descendents) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).BeginMark();
  }

  // Resolve all the values from the array
  for (const avro::GenericDatum& d : data) {
    // For all children
    for (const AvroParserSharedPtr& child : children) {
      TF_RETURN_IF_ERROR((*child).Parse(values, d, defaults));
    }
  }

  // Add a finish mark to all value buffers under this array
  for (const AvroParserSharedPtr& value_parser : final_descendents) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).FinishMark();
  }

  return OkStatus();
}
string ArrayAllParser::ToString(size_t level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---ArrayAllParser" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}

ArrayIndexParser::ArrayIndexParser(size_t index)
    : AvroParser(""), index_(index) {}
Status ArrayIndexParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
                               const avro::GenericDatum& datum,
                               const std::map<string, Tensor>& defaults) const {
  if (datum.type() != avro::AVRO_ARRAY) {
    return errors::InvalidArgument(
        TypeErrorMessage(GetSupportedTypes(), datum.type()));
  }

  // Check for valid index
  std::vector<avro::GenericDatum> data =
      datum.value<avro::GenericArray>().value();
  size_t n_elements = data.size();

  if (index_ > n_elements || index_ < 0) {
    return errors::InvalidArgument("Invalid index ", index_, ". Range [", 0,
                                   ", ", n_elements, ").");
  }
  // retrieve datum
  const avro::GenericDatum& d = data[index_];

  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  const std::vector<AvroParserSharedPtr>& final_descendents(
      GetFinalDescendents());

  // For all children same datum
  for (const AvroParserSharedPtr& child : children) {
    TF_RETURN_IF_ERROR((*child).Parse(values, d, defaults));
  }

  return OkStatus();
}
string ArrayIndexParser::ToString(size_t level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---ArrayIndexParser(" << index_ << ")"
     << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}

ArrayFilterParser::ArrayFilterParser(const tstring& lhs, const tstring& rhs,
                                     ArrayFilterType type)
    : AvroParser(""), lhs_(lhs), rhs_(rhs), type_(type) {}

ArrayFilterParser::ArrayFilterType ArrayFilterParser::ToArrayFilterType(
    bool lhs_is_constant, bool rhs_is_constant) {
  if (lhs_is_constant) {
    return kLhsIsConstant;
  }

  if (rhs_is_constant) {
    return kRhsIsConstant;
  }

  return kNoConstant;
}

Status ArrayFilterParser::Parse(
    std::map<string, ValueStoreUniquePtr>* values,
    const avro::GenericDatum& datum,
    const std::map<string, Tensor>& defaults) const {
  if (datum.type() != avro::AVRO_ARRAY) {
    return errors::InvalidArgument(
        TypeErrorMessage(GetSupportedTypes(), datum.type()));
  }

  const std::vector<AvroParserSharedPtr>& final_descendents =
      GetFinalDescendents();

  // Add a begin mark to all value buffers under this array
  for (const AvroParserSharedPtr& value_parser : final_descendents) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).BeginMark();
  }

  // Check for valid index
  std::vector<avro::GenericDatum> data =
      datum.value<avro::GenericArray>().value();
  size_t n_elements = data.size();

  const std::vector<AvroParserSharedPtr>& children(GetChildren());

  for (size_t i_elements = 0; i_elements < n_elements; ++i_elements) {
    size_t reverse_index = n_elements - i_elements;

    bool add_value = false;

    if (type_ == kRhsIsConstant) {
      add_value =
          (*(*values).at(lhs_)).ValueMatchesAtReverseIndex(rhs_, reverse_index);
    } else if (type_ == kLhsIsConstant) {
      add_value =
          (*(*values).at(rhs_)).ValueMatchesAtReverseIndex(lhs_, reverse_index);
    } else {
      add_value =
          (*(*values).at(lhs_))
              .ValuesMatchAtReverseIndex(*(*values).at(rhs_), reverse_index);
    }

    if (add_value) {
      const avro::GenericDatum& d = data[i_elements];
      // For all children
      for (const AvroParserSharedPtr& child : children) {
        TF_RETURN_IF_ERROR((*child).Parse(values, d, defaults));
      }
    }
  }

  // Add a finish mark to all value buffers under this array
  for (const AvroParserSharedPtr& value_parser : final_descendents) {
    // Assumes the key exists in the map
    (*(*values)[(*value_parser).GetKey()]).FinishMark();
  }

  return OkStatus();
}
string ArrayFilterParser::ToString(size_t level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---ArrayFilterParser(" << lhs_ << "=" << rhs_
     << ")" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}

MapKeyParser::MapKeyParser(const string& key) : AvroParser(""), key_(key) {}
Status MapKeyParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
                           const avro::GenericDatum& datum,
                           const std::map<string, Tensor>& defaults) const {
  if (datum.type() != avro::AVRO_MAP) {
    return errors::InvalidArgument(
        TypeErrorMessage(GetSupportedTypes(), datum.type()));
  }

  std::vector<std::pair<std::string, avro::GenericDatum>> data =
      datum.value<avro::GenericMap>().value();
  size_t n_elements = data.size();

  // TODO(fraudies): Optimize by caching in a map
  bool found = false;
  for (size_t i_elements = 0; i_elements < n_elements && !found; ++i_elements) {
    const auto& keyValue = data[i_elements];
    found = keyValue.first == key_;
    if (found) {
      // For all children
      const std::vector<AvroParserSharedPtr>& children(GetChildren());
      for (const AvroParserSharedPtr& child : children) {
        TF_RETURN_IF_ERROR((*child).Parse(values, keyValue.second, defaults));
      }
    }
  }

  if (!found) {
    return errors::InvalidArgument("Unable to find key '", key_, "'.");
  }

  return OkStatus();
}
string MapKeyParser::ToString(size_t level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---MapKeyParser(" << key_ << ")" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}

RecordParser::RecordParser(const string& name) : AvroParser(""), name_(name) {}
Status RecordParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
                           const avro::GenericDatum& datum,
                           const std::map<string, Tensor>& defaults) const {
  if (datum.type() != avro::AVRO_RECORD) {
    return errors::InvalidArgument(
        TypeErrorMessage(GetSupportedTypes(), datum.type()));
  }

  // Convert to record
  const avro::GenericRecord& record = datum.value<avro::GenericRecord>();

  // Return error if the field name does not exist
  if (!record.hasField(name_)) {
    return errors::InvalidArgument("Unable to find name '", name_, "'.");
  }

  // Get datum for field
  const avro::GenericDatum& d = record.field(name_);

  // Parse all children with this datum
  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  for (const AvroParserSharedPtr& child : children) {
    TF_RETURN_IF_ERROR((*child).Parse(values, d, defaults));
  }

  return OkStatus();
}
string RecordParser::ToString(size_t level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---RecordParser(" << name_ << ")"
     << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}

UnionParser::UnionParser(const string& type_name)
    : AvroParser(""), type_name_(type_name) {}
Status UnionParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
                          const avro::GenericDatum& datum,
                          const std::map<string, Tensor>& defaults) const {
  // Note, in this case we don't know the type
  // 2nd note, we don't need to resolve the branch, it's done already by the
  // read datum
  avro::Type datum_type = datum.type();

  // Find the right child for this type
  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  for (const AvroParserSharedPtr& child : children) {
    const std::set<avro::Type>& supported_types = (*child).GetSupportedTypes();
    // The child parser supports the data type, could be Null as well
    if (supported_types.find(datum_type) != supported_types.end()) {
      VLOG(5) << "For key '" << child->GetKey() << "' parse datum type '"
              << toString(datum_type) << "'.";
      TF_RETURN_IF_ERROR((*child).Parse(values, datum, defaults));
      // For any other type that is part of the branch resolve the null type
    } else if (supported_types.find(avro::AVRO_NULL) != supported_types.end()) {
      VLOG(5) << "For key '" << child->GetKey() << "' parse by using default";
      TF_RETURN_IF_ERROR(
          (*child).Parse(values, avro::GenericDatum(), defaults));
    }
  }
  // We could not find the right child, log a warning since this datum is lost
  // LOG(WARN) << "Branch value in data is not consumed";
  return OkStatus();
}
string UnionParser::ToString(size_t level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---UnionParser(" << type_name_ << ")"
     << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}

RootParser::RootParser() : AvroParser("") {}
Status RootParser::Parse(std::map<string, ValueStoreUniquePtr>* values,
                         const avro::GenericDatum& datum,
                         const std::map<string, Tensor>& defaults) const {
  const std::vector<AvroParserSharedPtr>& children(GetChildren());
  for (const AvroParserSharedPtr& child : children) {
    TF_RETURN_IF_ERROR((*child).Parse(values, datum, defaults));
  }
  return OkStatus();
}
string RootParser::ToString(size_t level) const {
  std::stringstream ss;
  ss << LevelToString(level) << "|---RootParser()" << std::endl;
  ss << ChildrenToString(level);
  return ss.str();
}

}  // namespace data
}  // namespace tensorflow
