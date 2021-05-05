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

#include "arrow/array.h"
#include "arrow/csv/reader.h"
#include "arrow/memory_pool.h"
#include "arrow/table.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow_io/arrow/kernels/arrow_kernels.h"
#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/core/kernels/io_stream.h"

namespace tensorflow {
namespace data {

class CSVReadable : public IOReadableInterface {
 public:
  CSVReadable(Env* env) : env_(env) {}

  ~CSVReadable() {}
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

    csv_file_.reset(new ArrowRandomAccessFile(file_.get(), file_size_));

    auto result = ::arrow::csv::TableReader::Make(
        ::arrow::default_memory_pool(), ::arrow::io::default_io_context(),
        csv_file_, ::arrow::csv::ReadOptions::Defaults(),
        ::arrow::csv::ParseOptions::Defaults(),
        ::arrow::csv::ConvertOptions::Defaults());
    if (!result.status().ok()) {
      return errors::InvalidArgument("unable to make a TableReader: ",
                                     result.status());
    }
    reader_ = std::move(result).ValueUnsafe();

    {
      auto result = reader_->Read();
      if (!result.status().ok()) {
        return errors::InvalidArgument("unable to read table: ",
                                       result.status());
      }
      table_ = std::move(result).ValueUnsafe();
    }

    for (int i = 0; i < table_->num_columns(); i++) {
      ::tensorflow::DataType dtype;
      switch (table_->column(i)->type()->id()) {
        case ::arrow::Type::BOOL:
          dtype = ::tensorflow::DT_BOOL;
          break;
        case ::arrow::Type::UINT8:
          dtype = ::tensorflow::DT_UINT8;
          break;
        case ::arrow::Type::INT8:
          dtype = ::tensorflow::DT_INT8;
          break;
        case ::arrow::Type::UINT16:
          dtype = ::tensorflow::DT_UINT16;
          break;
        case ::arrow::Type::INT16:
          dtype = ::tensorflow::DT_INT16;
          break;
        case ::arrow::Type::UINT32:
          dtype = ::tensorflow::DT_UINT32;
          break;
        case ::arrow::Type::INT32:
          dtype = ::tensorflow::DT_INT32;
          break;
        case ::arrow::Type::UINT64:
          dtype = ::tensorflow::DT_UINT64;
          break;
        case ::arrow::Type::INT64:
          dtype = ::tensorflow::DT_INT64;
          break;
        case ::arrow::Type::HALF_FLOAT:
          dtype = ::tensorflow::DT_HALF;
          break;
        case ::arrow::Type::FLOAT:
          dtype = ::tensorflow::DT_FLOAT;
          break;
        case ::arrow::Type::DOUBLE:
          dtype = ::tensorflow::DT_DOUBLE;
          break;
        case ::arrow::Type::STRING:
          dtype = ::tensorflow::DT_STRING;
          break;
        case ::arrow::Type::BINARY:
        case ::arrow::Type::FIXED_SIZE_BINARY:
        case ::arrow::Type::DATE32:
        case ::arrow::Type::DATE64:
        case ::arrow::Type::TIMESTAMP:
        case ::arrow::Type::TIME32:
        case ::arrow::Type::TIME64:
        case ::arrow::Type::DECIMAL:
        case ::arrow::Type::LIST:
        case ::arrow::Type::STRUCT:
        case ::arrow::Type::DICTIONARY:
        case ::arrow::Type::MAP:
        default:
          return errors::InvalidArgument("arrow data type is not supported: ",
                                         table_->column(i)->type()->ToString());
      }
      shapes_.push_back(TensorShape({static_cast<int64>(table_->num_rows())}));
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
    if (label) {
      *dtype = DT_BOOL;
    } else {
      *dtype = dtypes_[column_index];
    }
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

#define PROCESS_STRING_TYPE(ATYPE)                                \
  {                                                               \
    int64 curr_index = 0;                                         \
    for (auto chunk : slice->chunks()) {                          \
      for (int64_t item = 0; item < chunk->length(); item++) {    \
        value->flat<tstring>()(curr_index) =                      \
            (dynamic_cast<ATYPE*>(chunk.get()))->GetString(item); \
        curr_index++;                                             \
      }                                                           \
    }                                                             \
  }

    if (value != nullptr) {
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
        case DT_STRING:
          PROCESS_STRING_TYPE(::arrow::StringArray);
          break;
        default:
          return errors::InvalidArgument("data type is not supported: ",
                                         DataTypeString(value->dtype()));
      }
    }

    if (label != nullptr) {
      int64 curr_index = 0;
      for (auto chunk : slice->chunks()) {
        for (int64_t item = 0; item < chunk->length(); item++) {
          label->flat<bool>()(curr_index) = chunk->IsNull(item);
          curr_index++;
        }
      }
    }
    (*record_read) = element_stop - element_start;

    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("CSVReadable");
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ TF_GUARDED_BY(mu_);
  uint64 file_size_ TF_GUARDED_BY(mu_);
  std::shared_ptr<ArrowRandomAccessFile> csv_file_;
  std::shared_ptr<::arrow::csv::TableReader> reader_;
  std::shared_ptr<::arrow::Table> table_;

  std::vector<DataType> dtypes_;
  std::vector<TensorShape> shapes_;
  std::vector<string> columns_;
  std::unordered_map<string, int64> columns_index_;
};

REGISTER_KERNEL_BUILDER(Name("IO>CSVReadableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<CSVReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>CSVReadableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<CSVReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>CSVReadableRead").Device(DEVICE_CPU),
                        IOReadableReadOp<CSVReadable>);

}  // namespace data
}  // namespace tensorflow
