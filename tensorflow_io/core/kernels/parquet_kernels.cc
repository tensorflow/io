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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_io/core/kernels/io_interface.h"
#include "tensorflow_io/arrow/kernels/arrow_kernels.h"
#include "parquet/windows_compatibility.h"
#include "parquet/api/reader.h"


namespace tensorflow {
namespace data {
namespace {
class ParquetReadable : public IOReadableInterface {
 public:
  ParquetReadable(Env* env)
  : env_(env) {}

  ~ParquetReadable() {}
  Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) override {
    if (input.size() > 1) {
      return errors::InvalidArgument("more than 1 filename is not supported");
    }
    const string& filename = input[0];
    file_.reset(new SizedRandomAccessFile(env_, filename, memory_data, memory_size));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    parquet_file_.reset(new ArrowRandomAccessFile(file_.get(), file_size_));

    parquet_file_.reset(new ArrowRandomAccessFile(file_.get(), file_size_));
    parquet_reader_ = parquet::ParquetFileReader::Open(parquet_file_);
    parquet_metadata_ = parquet_reader_->metadata();

    shapes_.clear();
    dtypes_.clear();
    columns_.clear();
    for (size_t i = 0; i < parquet_metadata_->num_columns(); i++) {
      ::tensorflow::DataType dtype;
      switch(parquet_metadata_->schema()->Column(i)->physical_type()) {
      case parquet::Type::BOOLEAN:
        dtype = ::tensorflow::DT_BOOL;
        break;
      case parquet::Type::INT32:
        dtype = ::tensorflow::DT_INT32;
        break;
      case parquet::Type::INT64:
        dtype = ::tensorflow::DT_INT64;
        break;
      case parquet::Type::INT96: // Deprecated, thrown out exception when access with __getitem__
        dtype = ::tensorflow::DT_INT64;
        break;
      case parquet::Type::FLOAT:
        dtype = ::tensorflow::DT_FLOAT;
        break;
      case parquet::Type::DOUBLE:
        dtype = ::tensorflow::DT_DOUBLE;
        break;
      case parquet::Type::BYTE_ARRAY:
        dtype = ::tensorflow::DT_STRING;
        break;
      case parquet::Type::FIXED_LEN_BYTE_ARRAY:
        dtype = ::tensorflow::DT_STRING;
        break;
      default:
        return errors::InvalidArgument("parquet data type is not supported: ", parquet_metadata_->schema()->Column(i)->physical_type());
        break;
      }
      shapes_.push_back(TensorShape({static_cast<int64>(parquet_metadata_->num_rows())}));
      dtypes_.push_back(dtype);
      columns_.push_back(parquet_metadata_->schema()->Column(i)->path().get()->ToDotString());
      columns_index_[parquet_metadata_->schema()->Column(i)->path().get()->ToDotString()] = i;
    }

    return Status::OK();
  }
  Status Partitions(std::vector<int64> *partitions) override {
    partitions->clear();
    for (int row_group = 0; row_group < parquet_metadata_->num_row_groups(); row_group++) {
      std::shared_ptr<parquet::RowGroupReader> row_group_reader = parquet_reader_->RowGroup(row_group);
      partitions->push_back(row_group_reader->metadata()->num_rows());
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
  Status Spec(const string& component, PartialTensorShape* shape, DataType* dtype, bool label) override {
    if (columns_index_.find(component) == columns_index_.end()) {
      return errors::InvalidArgument("component ", component, " is invalid");
    }
    int64 column_index = columns_index_[component];
    *shape = shapes_[column_index];
    *dtype = dtypes_[column_index];
    return Status::OK();
  }

  Status Read(const int64 start, const int64 stop, const string& component, int64* record_read, Tensor* value, Tensor* label) override {
    if (columns_index_.find(component) == columns_index_.end()) {
      return errors::InvalidArgument("component ", component, " is invalid");
    }
    int64 column_index = columns_index_[component];
    (*record_read) = 0;
    if (start >= shapes_[column_index].dim_size(0)) {
      return Status::OK();
    }
    const string& column = component;
    int64 element_start = start < shapes_[column_index].dim_size(0) ? start : shapes_[column_index].dim_size(0);
    int64 element_stop = stop < shapes_[column_index].dim_size(0) ? stop : shapes_[column_index].dim_size(0);

    if (element_start > element_stop) {
      return errors::InvalidArgument("dataset ", column, " selection is out of boundary");
    }
    if (element_start == element_stop) {
      return Status::OK();
    }

    int64 row_group_offset = 0;
    for (int row_group = 0; row_group < parquet_metadata_->num_row_groups(); row_group++) {
      std::shared_ptr<parquet::RowGroupReader> row_group_reader = parquet_reader_->RowGroup(row_group);
      // Skip if row group is not within [start..stop]
      if ((row_group_offset + row_group_reader->metadata()->num_rows() < element_start) || (element_stop <= row_group_offset)) {
        row_group_offset += row_group_reader->metadata()->num_rows();
        continue;
      }
      // Find row_to_read range
      int64 row_to_read_start = row_group_offset > element_start ? row_group_offset : element_start;
      int64 row_to_read_final = (row_group_offset + row_group_reader->metadata()->num_rows()) < (element_stop) ? (row_group_offset + row_group_reader->metadata()->num_rows()) : (element_stop);
      int64 row_to_read_count = row_to_read_final - row_to_read_start;

      // TODO: parquet is RowGroup based so ideally the RowGroup should be cached
      // with the hope of indexing and slicing happens on each row. For now no caching
      // is done yet.
      std::shared_ptr<parquet::ColumnReader> column_reader = row_group_reader->Column(column_index);

      // buffer to fill location is value.data()[row_to_read_start - start]

      #define PARQUET_PROCESS_TYPE(ptype, type) { \
          parquet::TypedColumnReader<ptype>* reader = \
              static_cast<parquet::TypedColumnReader<ptype>*>( \
                  column_reader.get()); \
          if (row_to_read_start > row_group_offset) { \
            reader->Skip(row_to_read_start - row_group_offset); \
          } \
          ptype::c_type* value_p = (ptype::c_type *)(void *)(&(value->flat<type>().data()[row_to_read_start - element_start])); \
          int64_t values_read; \
          int64_t levels_read = reader->ReadBatch(row_to_read_count, nullptr, nullptr, value_p, &values_read); \
          if (!(levels_read == values_read && levels_read == row_to_read_count)) { \
            return errors::InvalidArgument("null value in column: ", column); \
          } \
        }

      #define PARQUET_PROCESS_BYTE_ARRAY(ptype) { \
          parquet::TypedColumnReader<ptype>* reader = \
              static_cast<parquet::TypedColumnReader<ptype>*>( \
                  column_reader.get()); \
          if (row_to_read_start > row_group_offset) { \
            reader->Skip(row_to_read_start - row_group_offset); \
          } \
          std::unique_ptr<ptype::c_type[]> value_p(new ptype::c_type[row_to_read_count]); \
          int64_t values_read; \
          int64_t levels_read = reader->ReadBatch(row_to_read_count, nullptr, nullptr, value_p.get(), &values_read); \
          if (!(levels_read == values_read && levels_read == row_to_read_count)) { \
            return errors::InvalidArgument("null value in column: ", column); \
          } \
          for (int64_t index = 0; index < values_read; index++) { \
            value->flat<tstring>()(row_to_read_start - element_start + index) = ByteArrayToString(value_p[index]); \
          } \
        }

      #define PARQUET_PROCESS_FIXED_LEN_BYTE_ARRAY(ptype, len) { \
          parquet::TypedColumnReader<ptype>* reader = \
              static_cast<parquet::TypedColumnReader<ptype>*>( \
                  column_reader.get()); \
          if (row_to_read_start > row_group_offset) { \
            reader->Skip(row_to_read_start - row_group_offset); \
          } \
          std::unique_ptr<ptype::c_type[]> value_p(new ptype::c_type[row_to_read_count]); \
          int64_t values_read; \
          int64_t levels_read = reader->ReadBatch(row_to_read_count, nullptr, nullptr, value_p.get(), &values_read); \
          if (!(levels_read == values_read && levels_read == row_to_read_count)) { \
            return errors::InvalidArgument("null value in column: ", column); \
          } \
          for (int64_t index = 0; index < values_read; index++) { \
            value->flat<tstring>()(row_to_read_start - element_start + index) = string((const char*)value_p[index].ptr, len); \
          } \
        }

      switch (parquet_metadata_->schema()->Column(column_index)->physical_type()) {
      case parquet::Type::BOOLEAN:
        PARQUET_PROCESS_TYPE(parquet::BooleanType, bool);
        break;
      case parquet::Type::INT32:
        PARQUET_PROCESS_TYPE(parquet::Int32Type, int32);
        break;
      case parquet::Type::INT64:
        PARQUET_PROCESS_TYPE(parquet::Int64Type, int64);
          break;
      case parquet::Type::FLOAT:
        PARQUET_PROCESS_TYPE(parquet::FloatType, float);
        break;
      case parquet::Type::DOUBLE:
        PARQUET_PROCESS_TYPE(parquet::DoubleType, double);
        break;
      case parquet::Type::BYTE_ARRAY:
        PARQUET_PROCESS_BYTE_ARRAY(parquet::ByteArrayType);
        break;
      case parquet::Type::FIXED_LEN_BYTE_ARRAY:
        PARQUET_PROCESS_FIXED_LEN_BYTE_ARRAY(parquet::FLBAType, parquet_metadata_->schema()->Column(column_index)->type_length());
        break;
      default:
        return errors::InvalidArgument("invalid data type: ", parquet_metadata_->schema()->Column(column_index)->physical_type());
      }
      row_group_offset += row_group_reader->metadata()->num_rows();
    }
    (*record_read) = element_stop - element_start;
    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("ParquetReadable");
  }
 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ TF_GUARDED_BY(mu_);
  uint64 file_size_ TF_GUARDED_BY(mu_);
  std::shared_ptr<ArrowRandomAccessFile> parquet_file_;
  std::unique_ptr<::parquet::ParquetFileReader> parquet_reader_;
  std::shared_ptr<::parquet::FileMetaData> parquet_metadata_;

  std::vector<DataType> dtypes_;
  std::vector<TensorShape> shapes_;
  std::vector<string> columns_;
  std::unordered_map<string, int64> columns_index_;
};

REGISTER_KERNEL_BUILDER(Name("IO>ParquetReadableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<ParquetReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>ParquetReadableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<ParquetReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>ParquetReadablePartitions").Device(DEVICE_CPU),
                        IOReadablePartitionsOp<ParquetReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>ParquetReadableRead").Device(DEVICE_CPU),
                        IOReadableReadOp<ParquetReadable>);

}  // namespace
}  // namespace data
}  // namespace tensorflow
