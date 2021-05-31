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

#include <fstream>
#include <iostream>

#include "arrow/array.h"
#include "arrow/json/reader.h"
#include "arrow/memory_pool.h"
#include "arrow/table.h"
#include "rapidjson/document.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_io/core/kernels/arrow/arrow_kernels.h"
#include "tensorflow_io/core/kernels/arrow/arrow_util.h"
#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/core/kernels/io_stream.h"

namespace tensorflow {
namespace data {
namespace {

class JSONReadable : public IOReadableInterface {
 public:
  JSONReadable(Env* env) : env_(env) {}

  ~JSONReadable() {}
  Status Init(const std::vector<string>& input,
              const std::vector<string>& metadata, const void* memory_data,
              const int64 memory_size) override {
    if (input.size() > 1) {
      return errors::InvalidArgument("more than 1 filename is not supported");
    }
    const string& filename = input[0];
    file_.reset(
        new SizedRandomAccessFile(env_, filename, memory_data, memory_size));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    mode_ = "ndjson";
    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find("mode: ") == 0) {
        mode_ = metadata[i].substr(6);
      }
    }

    if (mode_ == "records") {
      string buffer;
      buffer.resize(file_size_);
      StringPiece result;
      TF_RETURN_IF_ERROR(file_->Read(0, file_size_, &result, &buffer[0]));

      rapidjson::Document d;
      d.Parse(buffer.c_str());
      // Check the first element only
      const rapidjson::Value& a = d.GetArray();
      const rapidjson::Value& o = a[0];
      for (rapidjson::Value::ConstMemberIterator oi = o.MemberBegin();
           oi != o.MemberEnd(); ++oi) {
        DataType dtype;
        if (oi->value.IsInt64()) {
          dtype = DT_INT64;
        } else if (oi->value.IsDouble()) {
          dtype = DT_DOUBLE;
        } else {
          return errors::InvalidArgument("invalid data type: ",
                                         oi->name.GetString());
        }
        shapes_.push_back(TensorShape({static_cast<int64>(a.MemberCount())}));
        dtypes_.push_back(dtype);
        columns_.push_back(oi->name.GetString());
        columns_index_[oi->name.GetString()] =
            static_cast<int64>(columns_.size() - 1);
        tensors_.emplace_back(
            Tensor(dtype, TensorShape({static_cast<int64>(a.MemberCount())})));
      }
      // Fill in the values
      for (size_t i = 0; i < a.MemberCount(); i++) {
        const rapidjson::Value& o = a[i];
        for (size_t column_index = 0; column_index < columns_.size();
             column_index++) {
          const rapidjson::Value& v = o[columns_[column_index].c_str()];
          if (dtypes_[column_index] == DT_INT64) {
            tensors_[column_index].flat<int64>()(i) = v.GetInt64();
          } else if (dtypes_[column_index] == DT_DOUBLE) {
            tensors_[column_index].flat<double>()(i) = v.GetDouble();
          }
        }
      }

      return Status::OK();
    }

    json_file_.reset(new ArrowRandomAccessFile(file_.get(), file_size_));

    ::arrow::Status status;

    status = ::arrow::json::TableReader::Make(
        ::arrow::default_memory_pool(), json_file_,
        ::arrow::json::ReadOptions::Defaults(),
        ::arrow::json::ParseOptions::Defaults(), &reader_);
    if (!status.ok()) {
      return errors::InvalidArgument("unable to make a TableReader: ", status);
    }
    status = reader_->Read(&table_);
    if (!status.ok()) {
      return errors::InvalidArgument("unable to read table: ", status);
    }

    shapes_.clear();
    dtypes_.clear();
    columns_.clear();
    for (int i = 0; i < table_->num_columns(); i++) {
      shapes_.push_back(TensorShape({static_cast<int64>(table_->num_rows())}));
      ::tensorflow::DataType dtype;
      TF_RETURN_IF_ERROR(
          ArrowUtil::GetTensorFlowType(table_->column(i)->type(), &dtype));
      dtypes_.push_back(dtype);
      columns_.push_back(table_->ColumnNames()[i]);
      columns_index_[table_->ColumnNames()[i]] = i;
    }

    return Status::OK();
  }
  Status Components(std::vector<string>* components) override {
    components->clear();
    for (size_t i = 0; i < columns_.size(); i++) {
      components->push_back(columns_[i]);
    }
    return Status::OK();
  }
  Status Spec(const string& component, PartialTensorShape* shape,
              DataType* dtype, bool label) override {
    if (columns_index_.find(component) == columns_index_.end()) {
      return errors::InvalidArgument("component ", component, " is invalid");
    }
    int64 column_index = columns_index_[component];
    *shape = shapes_[column_index];
    *dtype = dtypes_[column_index];
    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop, const string& component,
              int64* record_read, Tensor* value, Tensor* label) override {
    if (columns_index_.find(component) == columns_index_.end()) {
      return errors::InvalidArgument("component ", component, " is invalid");
    }
    int64 column_index = columns_index_[component];

    (*record_read) = 0;
    if (start >= shapes_[column_index].dim_size(0)) {
      return Status::OK();
    }
    const string& column = component;
    int64 element_start = start < shapes_[column_index].dim_size(0)
                              ? start
                              : shapes_[column_index].dim_size(0);
    int64 element_stop = stop < shapes_[column_index].dim_size(0)
                             ? stop
                             : shapes_[column_index].dim_size(0);

    if (element_start > element_stop) {
      return errors::InvalidArgument("dataset ", column,
                                     " selection is out of boundary");
    }
    if (element_start == element_stop) {
      return Status::OK();
    }

    if (mode_ == "records") {
      if (dtypes_[column_index] == DT_INT64) {
        memcpy(&value->flat<int64>().data()[0],
               &tensors_[column_index].flat<int64>().data()[element_start],
               sizeof(int64) * (element_stop - element_start));
      } else if (dtypes_[column_index] == DT_DOUBLE) {
        memcpy(&value->flat<double>().data()[0],
               &tensors_[column_index].flat<double>().data()[element_start],
               sizeof(double) * (element_stop - element_start));
      } else {
        return errors::InvalidArgument("invalid data type: ",
                                       dtypes_[column_index]);
      }
      (*record_read) = element_stop - element_start;

      return Status::OK();
    }

    std::shared_ptr<::arrow::ChunkedArray> slice =
        table_->column(column_index)->Slice(element_start, element_stop);

#define PROCESS_TYPE(TTYPE, ATYPE)                             \
  {                                                            \
    int64 curr_index = 0;                                      \
    for (auto chunk : slice->chunks()) {                       \
      for (int64_t item = 0; item < chunk->length(); item++) { \
        value->flat<TTYPE>()(curr_index) =                     \
            (dynamic_cast<ATYPE*>(chunk.get()))->Value(item);  \
        curr_index++;                                          \
      }                                                        \
    }                                                          \
  }
    switch (value->dtype()) {
      case DT_BOOL:
        PROCESS_TYPE(bool, ::arrow::BooleanArray);
        break;
      case DT_INT8:
        PROCESS_TYPE(int8, ::arrow::NumericArray<::arrow::Int8Type>);
        break;
      case DT_UINT8:
        PROCESS_TYPE(uint8, ::arrow::NumericArray<::arrow::UInt8Type>);
        break;
      case DT_INT16:
        PROCESS_TYPE(int16, ::arrow::NumericArray<::arrow::Int16Type>);
        break;
      case DT_UINT16:
        PROCESS_TYPE(uint16, ::arrow::NumericArray<::arrow::UInt16Type>);
        break;
      case DT_INT32:
        PROCESS_TYPE(int32, ::arrow::NumericArray<::arrow::Int32Type>);
        break;
      case DT_UINT32:
        PROCESS_TYPE(uint32, ::arrow::NumericArray<::arrow::UInt32Type>);
        break;
      case DT_INT64:
        PROCESS_TYPE(int64, ::arrow::NumericArray<::arrow::Int64Type>);
        break;
      case DT_UINT64:
        PROCESS_TYPE(uint64, ::arrow::NumericArray<::arrow::UInt64Type>);
        break;
      case DT_FLOAT:
        PROCESS_TYPE(float, ::arrow::NumericArray<::arrow::FloatType>);
        break;
      case DT_DOUBLE:
        PROCESS_TYPE(double, ::arrow::NumericArray<::arrow::DoubleType>);
        break;
      default:
        return errors::InvalidArgument("data type is not supported: ",
                                       DataTypeString(value->dtype()));
    }
    (*record_read) = element_stop - element_start;
    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("JSONReadable");
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ TF_GUARDED_BY(mu_);
  uint64 file_size_ TF_GUARDED_BY(mu_);
  std::shared_ptr<ArrowRandomAccessFile> json_file_;
  std::shared_ptr<::arrow::json::TableReader> reader_;
  std::shared_ptr<::arrow::Table> table_;

  std::vector<Tensor> tensors_;
  string mode_;

  std::vector<DataType> dtypes_;
  std::vector<TensorShape> shapes_;
  std::vector<string> columns_;
  std::unordered_map<string, int64> columns_index_;
};

REGISTER_KERNEL_BUILDER(Name("IO>JSONReadableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<JSONReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>JSONReadableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<JSONReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>JSONReadableRead").Device(DEVICE_CPU),
                        IOReadableReadOp<JSONReadable>);

}  // namespace
}  // namespace data
}  // namespace tensorflow
