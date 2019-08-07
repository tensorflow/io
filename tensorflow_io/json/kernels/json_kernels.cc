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

#include <iostream>
#include <fstream>
#include "kernels/dataset_ops.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_io/core/kernels/stream.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "include/json/json.h"

namespace tensorflow {
namespace data {
namespace{

class ListJSONColumnsOp : public OpKernel {
public:
  explicit ListJSONColumnsOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string filename = filename_tensor.scalar<string>()();

    std::vector<string> columns;
    std::vector<string> dtypes;
    string error;

    // Read the whole JSON file to the memory.
    uint64 size = 0;
    OP_REQUIRES_OK(context, env_->GetFileSize(filename, &size));
    std::unique_ptr<tensorflow::RandomAccessFile> file;
    OP_REQUIRES_OK(context, env_->NewRandomAccessFile(filename, &file));
    string buffer_memory;
    StringPiece result;
    buffer_memory.resize(size);
    OP_REQUIRES_OK(context, file->Read(0, size, &result, &buffer_memory[0]));

    // Parse JSON records from the buffer string.
    Json::Reader reader;
    Json::Value records;
    OP_REQUIRES(context, reader.parse(buffer_memory, records), 
                errors::InvalidArgument("JSON parsing error: ", reader.getFormattedErrorMessages()));


    // Read one JSON record to get the list of the columns
    OP_REQUIRES(context, records.type() == Json::arrayValue,
                errors::InvalidArgument("JSON is not in record format!"));
    const Json::Value& record = records[0];
    columns = record.getMemberNames();
    for(size_t i=0; i<columns.size(); i++) {
      const string& column = columns[i];
      const Json::Value& val = record[column];
      string dtype;
      switch (val.type()){
      case Json::intValue:
        dtype = "int64";
        break;
      case Json::uintValue:
        dtype = "uint64";
        break;
      case Json::realValue:
        dtype = "double";
        break;
      case Json::stringValue:
        dtype = "string";
        break;
      case Json::booleanValue:
        dtype = "bool";
        break;
        //Currently Json::nullValue, Json::arrayValue, and Json::objectValue are not supported.
      default:
        break;
      }
      if (dtype == "") {
        continue;
      }
      dtypes.emplace_back(dtype);
    }
    TensorShape output_shape = filename_tensor.shape();
    output_shape.AddDim(columns.size());

    Tensor* columns_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &columns_tensor));
    Tensor* dtypes_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &dtypes_tensor));

    output_shape.AddDim(1);

    for (size_t i = 0; i < columns.size(); i++) {
      columns_tensor->flat<string>()(i) = columns[i];
      dtypes_tensor->flat<string>()(i) = dtypes[i];
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class ReadJSONOp : public OpKernel {
public:
  explicit ReadJSONOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string& filename = filename_tensor.scalar<string>()();

    const Tensor& column_tensor = context->input(1);
    const string& column = column_tensor.scalar<string>()();

    // Read the whole JSON file to the memory.
    uint64 size = 0;
    OP_REQUIRES_OK(context, env_->GetFileSize(filename, &size));
    std::unique_ptr<tensorflow::RandomAccessFile> file;
    OP_REQUIRES_OK(context, env_->NewRandomAccessFile(filename, &file));
    string buffer_memory;
    StringPiece result;
    buffer_memory.resize(size);
    OP_REQUIRES_OK(context, file->Read(0, size, &result, &buffer_memory[0]));


    // Parse JSON records from the buffer string.
    Json::Reader reader;
    Json::Value json_records;
    OP_REQUIRES(context, reader.parse(buffer_memory, json_records), 
                errors::InvalidArgument("JSON parsing error: ", reader.getFormattedErrorMessages()));

    OP_REQUIRES(context, json_records.type() == Json::arrayValue,
                errors::InvalidArgument("JSON is not in record format!"));

    #define BOOL_VALUE records.push_back(val.asBool())
    #define INT64_VALUE records.emplace_back(val.asInt64())
    #define UINT64_VALUE records.emplace_back(val.asUInt64())
    #define DOUBLE_VALUE records.emplace_back(val.asDouble())
    #define STRING_VALUE records.emplace_back(val.asString())

    #define PROCESS_RECORD(TYPE, VALUE) { \
      std::vector<TYPE> records; \
      for(Json::ArrayIndex i = 0; i < json_records.size(); i++) { \
        const Json::Value& record = json_records[i]; \
        const Json::Value& val = record[column]; \
        VALUE; \
      } \
      Tensor* output_tensor; \
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({static_cast<int64>(records.size())}), &output_tensor)); \
      for (size_t i = 0; i < records.size(); i++) { \
        output_tensor->flat<TYPE>()(i) = std::move(records[i]); \
      } \
    }

    const Json::Value& first_record = json_records[0];
    const Json::Value& value = first_record[column];
    switch(value.type()) {
    case Json::intValue:
      PROCESS_RECORD(int64, INT64_VALUE);
      break;
    case Json::uintValue:
      PROCESS_RECORD(uint64, UINT64_VALUE);
      break;
    case Json::realValue:
      PROCESS_RECORD(double, DOUBLE_VALUE);
      break;
    case Json::stringValue:
      PROCESS_RECORD(string, STRING_VALUE);
      break;
    case Json::booleanValue:
      PROCESS_RECORD(bool, BOOL_VALUE);
      break;
      //Currently Json::nullValue, Json::arrayValue, and Json::objectValue are not supported.
    default:
      OP_REQUIRES(context, false, errors::InvalidArgument("unsupported data type: ", value.type()));
    }
      
  }
private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("ListJSONColumns").Device(DEVICE_CPU),
                        ListJSONColumnsOp);
REGISTER_KERNEL_BUILDER(Name("ReadJSON").Device(DEVICE_CPU),
                        ReadJSONOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
