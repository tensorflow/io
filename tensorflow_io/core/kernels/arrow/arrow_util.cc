/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow_io/core/kernels/arrow/arrow_util.h"

#include "arrow/adapters/tensorflow/convert.h"
#include "arrow/api.h"
#include "arrow/ipc/api.h"
#include "arrow/util/io_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {
namespace ArrowUtil {

Status GetTensorFlowType(std::shared_ptr<::arrow::DataType> dtype,
                         ::tensorflow::DataType* out) {
  if (dtype->id() == ::arrow::Type::STRING) {
    *out = ::tensorflow::DT_STRING;
    return Status::OK();
  }
  if (dtype->id() == ::arrow::Type::BINARY) {
    *out = ::tensorflow::DT_STRING;
    return Status::OK();
  }
  ::arrow::Status status =
      ::arrow::adapters::tensorflow::GetTensorFlowType(dtype, out);
  if (!status.ok()) {
    return errors::InvalidArgument("arrow data type ", dtype,
                                   " is not supported: ", status);
  }
  return Status::OK();
}

Status GetArrowType(::tensorflow::DataType dtype,
                    std::shared_ptr<::arrow::DataType>* out) {
  if (dtype == ::tensorflow::DT_STRING) {
    *out = ::arrow::utf8();
    return Status::OK();
  }
  ::arrow::Status status =
      ::arrow::adapters::tensorflow::GetArrowType(dtype, out);
  if (!status.ok()) {
    return errors::InvalidArgument("tensorflow data type ", dtype,
                                   " is not supported: ", status);
  }
  return Status::OK();
}

class ArrowAssignSpecImpl : public arrow::ArrayVisitor {
 public:
  ArrowAssignSpecImpl() : i_(0), batch_size_(0) {}

  Status AssignDataType(std::shared_ptr<arrow::Array> array,
                        ::tensorflow::DataType* out_dtype) {
    return AssignSpec(array, 0, 0, out_dtype, nullptr);
  }

  Status AssignShape(std::shared_ptr<arrow::Array> array, int64 i,
                     int64 batch_size, TensorShape* out_shape) {
    return AssignSpec(array, i, batch_size, nullptr, out_shape);
  }

  // Get the DataType and equivalent TensorShape for a given Array, taking into
  // account possible batch size
  Status AssignSpec(std::shared_ptr<arrow::Array> array, int64 i,
                    int64 batch_size, ::tensorflow::DataType* out_dtype,
                    TensorShape* out_shape) {
    i_ = i;
    batch_size_ = batch_size;
    out_shape_ = out_shape;
    out_dtype_ = out_dtype;

    // batch_size of 0 indicates 1 record at a time, no batching
    if (batch_size_ > 0) {
      out_shape_->AddDim(batch_size_);
    }

    CHECK_ARROW(array->Accept(this));
    return Status::OK();
  }

 protected:
  template <typename ArrayType>
  arrow::Status VisitPrimitive(const ArrayType& array) {
    if (out_dtype_ != nullptr) {
      return ::arrow::adapters::tensorflow::GetTensorFlowType(array.type(),
                                                              out_dtype_);
    }
    return arrow::Status::OK();
  }

#define VISIT_PRIMITIVE(TYPE)                               \
  virtual arrow::Status Visit(const TYPE& array) override { \
    return VisitPrimitive(array);                           \
  }

  VISIT_PRIMITIVE(arrow::BooleanArray)
  VISIT_PRIMITIVE(arrow::Int8Array)
  VISIT_PRIMITIVE(arrow::Int16Array)
  VISIT_PRIMITIVE(arrow::Int32Array)
  VISIT_PRIMITIVE(arrow::Int64Array)
  VISIT_PRIMITIVE(arrow::UInt8Array)
  VISIT_PRIMITIVE(arrow::UInt16Array)
  VISIT_PRIMITIVE(arrow::UInt32Array)
  VISIT_PRIMITIVE(arrow::UInt64Array)
  VISIT_PRIMITIVE(arrow::HalfFloatArray)
  VISIT_PRIMITIVE(arrow::FloatArray)
  VISIT_PRIMITIVE(arrow::DoubleArray)
  VISIT_PRIMITIVE(arrow::StringArray)
  VISIT_PRIMITIVE(arrow::BinaryArray)
#undef VISIT_PRIMITIVE

  virtual arrow::Status Visit(const arrow::ListArray& array) override {
    int32 values_offset = array.value_offset(i_);
    int32 array_length = array.value_length(i_);
    int32 num_arrays = 1;

    // If batching tensors, arrays must be same length
    if (batch_size_ > 0) {
      num_arrays = batch_size_;
      for (int64 j = i_; j < i_ + num_arrays; ++j) {
        if (array.value_length(j) != array_length) {
          return arrow::Status::Invalid(
              "Batching variable-length arrays is unsupported");
        }
      }
    }

    // Add diminsion for array
    if (out_shape_ != nullptr) {
      out_shape_->AddDim(array_length);
    }

    // Prepare the array data buffer and visit the array slice
    std::shared_ptr<arrow::Array> values = array.values();
    std::shared_ptr<arrow::Array> element_values =
        values->Slice(values_offset, array_length * num_arrays);
    return element_values->Accept(this);
  }

 private:
  int64 i_;
  int64 batch_size_;
  DataType* out_dtype_;
  TensorShape* out_shape_;
};

Status AssignShape(std::shared_ptr<arrow::Array> array, int64 i,
                   int64 batch_size, TensorShape* out_shape) {
  ArrowAssignSpecImpl visitor;
  return visitor.AssignShape(array, i, batch_size, out_shape);
}

Status AssignSpec(std::shared_ptr<arrow::Array> array, int64 i,
                  int64 batch_size, ::tensorflow::DataType* out_dtype,
                  TensorShape* out_shape) {
  ArrowAssignSpecImpl visitor;
  return visitor.AssignSpec(array, i, batch_size, out_dtype, out_shape);
}

// Assign elements of an Arrow Array to a Tensor
class ArrowAssignTensorImpl : public arrow::ArrayVisitor {
 public:
  ArrowAssignTensorImpl() : i_(0), out_tensor_(nullptr) {}

  Status AssignTensor(std::shared_ptr<arrow::Array> array, int64 i,
                      Tensor* out_tensor) {
    i_ = i;
    out_tensor_ = out_tensor;
    if (array->null_count() != 0) {
      return errors::Internal(
          "Arrow arrays with null values not currently supported");
    }
    CHECK_ARROW(array->Accept(this));
    return Status::OK();
  }

 protected:
  virtual arrow::Status Visit(const arrow::BooleanArray& array) {
    // Must copy one value at a time because Arrow stores values as bits
    auto shape = out_tensor_->shape();
    for (int64 j = 0; j < shape.num_elements(); ++j) {
      // NOTE: for Array ListArray, curr_row_idx_ is 0 for element array
      bool value = array.Value(i_ + j);
      void* dst = const_cast<char*>(out_tensor_->tensor_data().data()) +
                  j * sizeof(value);
      memcpy(dst, &value, sizeof(value));
    }

    return arrow::Status::OK();
  }

  template <typename ArrayType>
  arrow::Status VisitFixedWidth(const ArrayType& array) {
    const auto& fw_type =
        static_cast<const arrow::FixedWidthType&>(*array.type());
    const int64_t type_width = fw_type.bit_width() / 8;

    // TODO: verify tensor is correct shape, arrow array is within bounds

    // Primitive Arrow arrays have validity and value buffers, currently
    // only arrays with null count == 0 are supported, so only need values here
    static const int VALUE_BUFFER = 1;
    auto values = array.data()->buffers[VALUE_BUFFER];
    if (values == NULLPTR) {
      return arrow::Status::Invalid(
          "Received an Arrow array with a NULL value buffer");
    }

    const void* src =
        (values->data() + array.data()->offset * type_width) + i_ * type_width;
    void* dst = const_cast<char*>(out_tensor_->tensor_data().data());
    std::memcpy(dst, src, out_tensor_->NumElements() * type_width);

    return arrow::Status::OK();
  }

#define VISIT_FIXED_WIDTH(TYPE)                             \
  virtual arrow::Status Visit(const TYPE& array) override { \
    return VisitFixedWidth(array);                          \
  }

  VISIT_FIXED_WIDTH(arrow::Int8Array)
  VISIT_FIXED_WIDTH(arrow::Int16Array)
  VISIT_FIXED_WIDTH(arrow::Int32Array)
  VISIT_FIXED_WIDTH(arrow::Int64Array)
  VISIT_FIXED_WIDTH(arrow::UInt8Array)
  VISIT_FIXED_WIDTH(arrow::UInt16Array)
  VISIT_FIXED_WIDTH(arrow::UInt32Array)
  VISIT_FIXED_WIDTH(arrow::UInt64Array)
  VISIT_FIXED_WIDTH(arrow::HalfFloatArray)
  VISIT_FIXED_WIDTH(arrow::FloatArray)
  VISIT_FIXED_WIDTH(arrow::DoubleArray)
#undef VISIT_FIXED_WITH

  virtual arrow::Status Visit(const arrow::ListArray& array) override {
    int32 values_offset = array.value_offset(i_);
    int32 curr_array_length = array.value_length(i_);
    int32 num_arrays = 1;
    auto shape = out_tensor_->shape();

    // If batching tensors, arrays must be same length
    if (shape.dims() > 1) {
      num_arrays = shape.dim_size(0);
      for (int64_t j = i_; j < i_ + num_arrays; ++j) {
        if (array.value_length(j) != curr_array_length) {
          return arrow::Status::Invalid(
              "Batching variable-length arrays is unsupported");
        }
      }
    }

    // Save current index and swap after array is copied
    int32 tmp_index = i_;
    i_ = 0;

    // Prepare the array data buffer and visit the array slice
    std::shared_ptr<arrow::Array> values = array.values();
    std::shared_ptr<arrow::Array> element_values =
        values->Slice(values_offset, curr_array_length * num_arrays);
    auto result = element_values->Accept(this);

    // Reset state variables for next time
    i_ = tmp_index;
    return result;
  }

  virtual arrow::Status Visit(const arrow::StringArray& array) override {
    auto shape = out_tensor_->shape();
    auto output_flat = out_tensor_->flat<tstring>();

    for (int64 j = 0; j < shape.num_elements(); ++j) {
      output_flat(j) = array.GetString(i_ + j);
    }

    return arrow::Status::OK();
  }

  virtual arrow::Status Visit(const arrow::BinaryArray& array) override {
    auto shape = out_tensor_->shape();
    auto output_flat = out_tensor_->flat<tstring>();

    for (int64 j = 0; j < shape.num_elements(); ++j) {
      output_flat(j) = array.GetString(i_ + j);
    }

    return arrow::Status::OK();
  }

 private:
  int64 i_;
  int32 curr_array_length_;
  Tensor* out_tensor_;
};

Status AssignTensor(std::shared_ptr<arrow::Array> array, int64 i,
                    Tensor* out_tensor) {
  ArrowAssignTensorImpl visitor;
  return visitor.AssignTensor(array, i, out_tensor);
}

// Check the type of an Arrow array matches expected tensor type
class ArrowArrayTypeCheckerImpl : public arrow::TypeVisitor {
 public:
  Status CheckArrayType(std::shared_ptr<arrow::DataType> type,
                        ::tensorflow::DataType expected_type) {
    expected_type_ = expected_type;

    // First see if complex type handled by visitor
    arrow::Status visit_status = type->Accept(this);
    if (visit_status.ok()) {
      return Status::OK();
    }

    // Check type as a scalar type
    CHECK_ARROW(CheckScalarType(type));
    return Status::OK();
  }

 protected:
  virtual arrow::Status Visit(const arrow::ListType& type) {
    return CheckScalarType(type.value_type());
  }

  // Check scalar types with arrow::adapters::tensorflow
  arrow::Status CheckScalarType(std::shared_ptr<arrow::DataType> scalar_type) {
    DataType converted_type;
    ::tensorflow::Status status =
        GetTensorFlowType(scalar_type, &converted_type);
    if (!status.ok()) {
      return ::arrow::Status::Invalid(status);
    }
    if (converted_type != expected_type_) {
      return arrow::Status::TypeError(
          "Arrow type mismatch: expected dtype=" +
          std::to_string(expected_type_) +
          ", but got dtype=" + std::to_string(converted_type));
    }
    return arrow::Status::OK();
  }

 private:
  DataType expected_type_;
};

Status CheckArrayType(std::shared_ptr<arrow::DataType> type,
                      ::tensorflow::DataType expected_type) {
  ArrowArrayTypeCheckerImpl visitor;
  return visitor.CheckArrayType(type, expected_type);
}

class ArrowMakeArrayDataImpl : public arrow::TypeVisitor {
 public:
  Status Make(std::shared_ptr<arrow::DataType> type,
              std::vector<int64> array_lengths,
              std::vector<std::shared_ptr<arrow::Buffer>> buffers,
              std::shared_ptr<arrow::ArrayData>* out_data) {
    type_ = type;
    lengths_ = array_lengths;
    buffers_ = buffers;
    out_data_ = out_data;
    CHECK_ARROW(type->Accept(this));
    return Status::OK();
  }

 protected:
  template <typename DataTypeType>
  arrow::Status VisitPrimitive(const DataTypeType& type) {
    // TODO null count == 0
    *out_data_ =
        arrow::ArrayData::Make(type_, lengths_[0], std::move(buffers_), 0);
    return arrow::Status::OK();
  }

#define VISIT_PRIMITIVE(TYPE)                              \
  virtual arrow::Status Visit(const TYPE& type) override { \
    return VisitPrimitive(type);                           \
  }

  VISIT_PRIMITIVE(arrow::BooleanType)
  VISIT_PRIMITIVE(arrow::Int8Type)
  VISIT_PRIMITIVE(arrow::Int16Type)
  VISIT_PRIMITIVE(arrow::Int32Type)
  VISIT_PRIMITIVE(arrow::Int64Type)
  VISIT_PRIMITIVE(arrow::UInt8Type)
  VISIT_PRIMITIVE(arrow::UInt16Type)
  VISIT_PRIMITIVE(arrow::UInt32Type)
  VISIT_PRIMITIVE(arrow::UInt64Type)
  VISIT_PRIMITIVE(arrow::HalfFloatType)
  VISIT_PRIMITIVE(arrow::FloatType)
  VISIT_PRIMITIVE(arrow::DoubleType)
#undef VISIT_PRIMITIVE

  virtual arrow::Status Visit(const arrow::ListType& type) override {
    // TODO assert buffers and lengths size

    // Copy first 2 buffers, which are validity and offset buffers for the list
    std::vector<std::shared_ptr<arrow::Buffer>> list_bufs(buffers_.begin(),
                                                          buffers_.begin() + 2);
    buffers_.erase(buffers_.begin(), buffers_.begin() + 2);

    // Copy first array length for length of list
    int64 list_length = lengths_[0];
    lengths_.erase(lengths_.begin(), lengths_.begin() + 1);

    // Make array data for the child type
    type_ = type.value_type();
    type.value_type()->Accept(this);
    auto child_data = *out_data_;

    // Make array data for the list TODO null count == 0
    auto list_type = std::make_shared<arrow::ListType>(type.value_type());
    *out_data_ = arrow::ArrayData::Make(list_type, list_length,
                                        std::move(list_bufs), {child_data}, 0);

    return arrow::Status::OK();
  }

 private:
  std::shared_ptr<arrow::DataType> type_;
  std::vector<std::shared_ptr<arrow::Buffer>> buffers_;
  std::vector<int64> lengths_;
  std::shared_ptr<arrow::ArrayData>* out_data_;
};

Status MakeArrayData(std::shared_ptr<arrow::DataType> type,
                     std::vector<int64> array_lengths,
                     std::vector<std::shared_ptr<arrow::Buffer>> buffers,
                     std::shared_ptr<arrow::ArrayData>* out_data) {
  ArrowMakeArrayDataImpl visitor;
  return visitor.Make(type, array_lengths, buffers, out_data);
}

Status ParseEndpoint(std::string endpoint, std::string* endpoint_type,
                     std::string* endpoint_value) {
  size_t sep_pos = endpoint.find(':');

  // Check for a proper format
  if (sep_pos == std::string::npos) {
    return errors::InvalidArgument(
        "Expected endpoint to be in format <endpoint_type>://<endpoint_value> "
        "or <host>:<port> for tcp IPv4, but got: " +
        endpoint);
  }

  // If IPv4 and no endpoint type specified, descriptor is entire endpoint
  if (endpoint.substr(sep_pos + 1, 2) != "//") {
    *endpoint_type = "";
    *endpoint_value = endpoint;
    return Status::OK();
  }

  // Parse string as <endpoint_type>://<endpoint_value>
  *endpoint_type = endpoint.substr(0, sep_pos);
  *endpoint_value = endpoint.substr(sep_pos + 3);

  return Status::OK();
}

Status ParseHost(std::string host, std::string* host_address,
                 std::string* host_port) {
  size_t sep_pos = host.find(':');
  if (sep_pos == std::string::npos || sep_pos == host.length()) {
    return errors::InvalidArgument(
        "Expected host to be in format <host>:<port> but got: " + host);
  }

  *host_address = host.substr(0, sep_pos);
  *host_port = host.substr(sep_pos + 1);

  return Status::OK();
}

enum calType {
  CONSTANT,
  VARIABLE,
  ADD,
  SUBTRACT,
  MULTIPLY,
  DIVIDE,
  EQUAL,
  NOT_EQUAL,
  LESS,
  LESS_EQUAL,
  GREATER,
  GREATER_EQUAL,
  AND,
  OR,
  LPAREN,
  RPAREN,
};

enum OpType {
  OPERATOR,
  OPERAND,
};

typedef struct Token {
  Token(calType type, int value) : type_(type) {
    if (type_ == calType::CONSTANT) {
      expression_ = arrow::compute::literal(value);
    }
  }
  Token(calType type, float value) : type_(type) {
    if (type_ == calType::CONSTANT) {
      expression_ = arrow::compute::literal(value);
    }
  }
  Token(calType type, std::string func) : type_(type), func_(func) {
    if (type_ == calType::VARIABLE) {
      expression_ = arrow::compute::field_ref(func_);
    }
  }
  calType type_;
  std::string func_;
  arrow::compute::Expression expression_;
} Token;

typedef struct ASTNode {
  ASTNode(std::shared_ptr<Token> token, std::shared_ptr<ASTNode> left,
          std::shared_ptr<ASTNode> right)
      : token_(token), left_(left), right_(right) {}
  std::shared_ptr<Token> token_;
  std::shared_ptr<ASTNode> left_;
  std::shared_ptr<ASTNode> right_;
} ASTNode;

class Lexer {
 public:
  Lexer(const std::string& text)
      : text_(text), position_(0), cur_op_(OPERATOR){};
  void skip_space() {
    while (position_ < text_.length() && text_[position_] == ' ') {
      position_++;
    }
  }

  std::string get_constant() {
    int start = position_ - 1;
    while (position_ < text_.length() && std::isdigit(text_[position_]) ||
           '.' == text_[position_]) {
      position_++;
    }
    return text_.substr(start, position_ - start).c_str();
  }

  calType get_comparison_type() {
    // == != >= <=
    char begin_char = text_[position_ - 1];
    if (position_ < text_.length() && text_[position_] == '=') {
      position_++;
      if (begin_char == '=') {
        return calType::EQUAL;
      } else if (begin_char == '!') {
        return calType::NOT_EQUAL;
      } else if (begin_char == '>') {
        return calType::GREATER_EQUAL;
      } else if (begin_char == '<') {
        return calType::LESS_EQUAL;
      }
    } else {
      if (begin_char == '>') {
        return calType::GREATER;
      } else if (begin_char == '<') {
        return calType::LESS;
      }
    }
  }

  std::string get_variable() {
    int start = position_ - 1;
    while (position_ < text_.length() &&
           (std::isalnum(text_[position_]) || '_' == text_[position_])) {
      position_++;
    }
    return text_.substr(start, position_ - start);
  }

  std::shared_ptr<Token> get_next_token() {
    while (position_ < text_.length()) {
      char current_char = text_[position_++];
      if (' ' == current_char) {
        skip_space();
      } else if (std::isdigit(current_char)) {
        cur_op_ = OPERAND;
        std::string constant = get_constant();
        if (std::string::npos == constant.find('.')) {
          return std::make_shared<Token>(calType::CONSTANT,
                                         std::stoi(constant));
        } else {
          return std::make_shared<Token>(calType::CONSTANT,
                                         std::stof(constant));
        }
      } else if (std::isalpha(current_char) || '_' == current_char) {
        cur_op_ = OPERAND;
        return std::make_shared<Token>(calType::VARIABLE, get_variable());
      } else if ('+' == current_char) {
        cur_op_ = OPERATOR;
        return std::make_shared<Token>(calType::ADD, "add");
      } else if ('-' == current_char) {
        if (cur_op_ == OPERAND) {
          cur_op_ = OPERATOR;
          return std::make_shared<Token>(calType::SUBTRACT, "subtract");
        } else {
          cur_op_ = OPERAND;
          std::string constant = get_constant();
          if (constant.length() <= 1) {
            return nullptr;
          }
          if (std::string::npos == constant.find('.')) {
            return std::make_shared<Token>(calType::CONSTANT,
                                           std::stoi(constant));
          } else {
            return std::make_shared<Token>(calType::CONSTANT,
                                           std::stof(constant));
          }
        }
      } else if ('*' == current_char) {
        cur_op_ = OPERATOR;
        return std::make_shared<Token>(calType::MULTIPLY, "multiply");
      } else if ('/' == current_char) {
        cur_op_ = OPERATOR;
        return std::make_shared<Token>(calType::DIVIDE, "divide");
      } else if ('(' == current_char) {
        cur_op_ = OPERATOR;
        return std::make_shared<Token>(calType::LPAREN, "(");
      } else if (')' == current_char) {
        cur_op_ = OPERAND;
        return std::make_shared<Token>(calType::RPAREN, ")");
      } else if ('=' == current_char || '!' == current_char ||
                 '>' == current_char || '<' == current_char) {
        cur_op_ = OPERATOR;
        auto type = get_comparison_type();
        if (calType::EQUAL == type) {
          return std::make_shared<Token>(calType::EQUAL, "equal");
        } else if (calType::NOT_EQUAL == type) {
          return std::make_shared<Token>(calType::NOT_EQUAL, "not_equal");
        } else if (calType::LESS == type) {
          return std::make_shared<Token>(calType::LESS, "less");
        } else if (calType::LESS_EQUAL == type) {
          return std::make_shared<Token>(calType::LESS_EQUAL, "less_equal");
        } else if (calType::GREATER == type) {
          return std::make_shared<Token>(calType::GREATER, "greater");
        } else if (calType::GREATER_EQUAL == type) {
          return std::make_shared<Token>(calType::GREATER_EQUAL,
                                         "greater_equal");
        }
      } else if ('&' == current_char) {
        cur_op_ = OPERATOR;
        if (position_ < text_.length() && '&' == text_[position_]) {
          position_++;
          return std::make_shared<Token>(calType::AND, "and");
        }
      } else if ('|' == current_char) {
        cur_op_ = OPERATOR;
        if (position_ < text_.length() && '|' == text_[position_]) {
          position_++;
          return std::make_shared<Token>(calType::OR, "or");
        }
      }
    }
    return nullptr;
  }

 private:
  OpType cur_op_;
  int position_;
  std::string text_;
};

class Parser {
 public:
  Parser(std::shared_ptr<Lexer> ptr) : lexer_ptr_(ptr) {
    current_token_ = lexer_ptr_->get_next_token();
  }

  inline void update_current_token() {
    current_token_ = lexer_ptr_->get_next_token();
  }

  // constant, variable, lparen
  std::shared_ptr<ASTNode> factor() {
    if (!current_token_) {
      return nullptr;
    }
    auto token = current_token_;
    if (token->type_ == calType::CONSTANT) {
      update_current_token();
      return std::make_shared<ASTNode>(token, nullptr, nullptr);
    } else if (token->type_ == calType::VARIABLE) {
      update_current_token();
      return std::make_shared<ASTNode>(token, nullptr, nullptr);
    } else if (token->type_ == calType::LPAREN) {
      update_current_token();
      auto node = logical();
      update_current_token();
      return node;
    }
    return nullptr;
  }

  // multiply, divide
  std::shared_ptr<ASTNode> term() {
    auto node = factor();
    while (current_token_ && (current_token_->type_ == calType::MULTIPLY ||
                              current_token_->type_ == calType::DIVIDE)) {
      auto token = current_token_;
      update_current_token();
      node = std::make_shared<ASTNode>(token, node, factor());
    }
    return node;
  }

  // add, subtract
  std::shared_ptr<ASTNode> expr() {
    auto node = term();
    while (current_token_ && (current_token_->type_ == calType::ADD ||
                              current_token_->type_ == calType::SUBTRACT)) {
      auto token = current_token_;
      update_current_token();
      node = std::make_shared<ASTNode>(token, node, term());
    }
    return node;
  }

  // Comparison
  std::shared_ptr<ASTNode> comparison() {
    auto node = expr();
    while (current_token_ &&
           (current_token_->type_ >= calType::EQUAL &&
            current_token_->type_ <= calType::GREATER_EQUAL)) {
      auto token = current_token_;
      update_current_token();
      node = std::make_shared<ASTNode>(token, node, expr());
    }
    return node;
  }

  // Logical
  std::shared_ptr<ASTNode> logical() {
    auto node = comparison();
    while (current_token_ && (current_token_->type_ == calType::AND ||
                              current_token_->type_ == calType::OR)) {
      auto token = current_token_;
      update_current_token();
      node = std::make_shared<ASTNode>(token, node, comparison());
    }
    return node;
  }

 private:
  std::shared_ptr<Lexer> lexer_ptr_;
  std::shared_ptr<Token> current_token_;
};

class Interpreter {
 public:
  Interpreter(std::shared_ptr<Parser> parser) : parser_(parser) {}
  arrow::compute::Expression visit(std::shared_ptr<ASTNode> root) {
    auto rt = root->token_;
    auto rlt = root->left_->token_;
    auto rrt = root->right_->token_;
    if (rlt->type_ != calType::CONSTANT && rlt->type_ != calType::VARIABLE) {
      visit(root->left_);
    }
    if (rrt->type_ != calType::CONSTANT && rrt->type_ != calType::VARIABLE) {
      visit(root->right_);
    }

    if (rt->type_ >= calType::ADD && rt->type_ <= calType::OR) {
      rt->expression_ =
          arrow::compute::call(rt->func_, {rlt->expression_, rrt->expression_});
    }
    rt->type_ = calType::VARIABLE;
    return rt->expression_;
  }

  Status interpreter(std::shared_ptr<ASTNode>& ASTree) {
    auto root = parser_->logical();
    if (!root || !root->left_ || !root->right_ ||
        root->token_->type_ < calType::EQUAL ||
        root->token_->type_ > calType::OR) {
      return errors::InvalidArgument(
          "Your filter expression is not supported!");
    }
    ASTree = root;
    return Status::OK();
  }

 private:
  std::shared_ptr<Parser> parser_;
};

Status ParseExpression(const std::string& text,
                       arrow::compute::Expression& expr) {
  auto lexer_ptr = std::make_shared<Lexer>(text);
  auto parser_ptr = std::make_shared<Parser>(lexer_ptr);
  auto interpreter_ptr = std::make_shared<Interpreter>(parser_ptr);

  std::shared_ptr<ASTNode> ASTree;
  auto status = interpreter_ptr->interpreter(ASTree);
  if (!status.ok()) {
    return status;
  }

  expr = interpreter_ptr->visit(ASTree);
  return Status::OK();
}

}  // namespace ArrowUtil
}  // namespace data
}  // namespace tensorflow
