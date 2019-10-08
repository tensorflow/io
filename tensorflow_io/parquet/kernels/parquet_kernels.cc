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
#include "parquet/api/reader.h"

namespace tensorflow {
namespace data {
namespace {

class ListParquetColumnsOp : public OpKernel {
 public:
  explicit ListParquetColumnsOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string filename = filename_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(1);
    const string& memory = memory_tensor.scalar<string>()();

    std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory.data(), memory.size()));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    std::shared_ptr<ArrowRandomAccessFile> parquet_file(new ArrowRandomAccessFile(file.get(), size));
    std::shared_ptr<::parquet::FileMetaData> metadata = ::parquet::ReadMetaData(parquet_file);

    std::vector<string> columns;
    std::vector<string> dtypes;
    std::vector<int64> counts;
    columns.reserve(metadata->num_columns());
    dtypes.reserve(metadata->num_columns());
    counts.reserve(metadata->num_columns());
    for (int i = 0; i < metadata->num_columns(); i++) {
      string dtype = "";
      switch(metadata->schema()->Column(i)->physical_type()) {
      case parquet::Type::BOOLEAN:
        dtype = "bool";
        break;
      case parquet::Type::INT32:
        dtype = "int32";
        break;
      case parquet::Type::INT64:
        dtype = "int64";
        break;
      case parquet::Type::FLOAT:
        dtype = "float";
        break;
      case parquet::Type::DOUBLE:
        dtype = "double";
        break;
      default:
        // Unsupported data type INT96, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY
        break;
      }
      if (dtype == "") {
        continue;
      }
      columns.push_back(metadata->schema()->Column(i)->path().get()->ToDotString());
      dtypes.push_back(dtype);
      counts.push_back(metadata->num_rows());
    }

    TensorShape output_shape = filename_tensor.shape();
    output_shape.AddDim(columns.size());

    Tensor* columns_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &columns_tensor));
    Tensor* dtypes_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &dtypes_tensor));

    output_shape.AddDim(1);

    Tensor* shapes_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(2, output_shape, &shapes_tensor));

    for (int i = 0; i < columns.size(); i++) {
      columns_tensor->flat<string>()(i) = columns[i];
      dtypes_tensor->flat<string>()(i) = dtypes[i];
      shapes_tensor->flat<int64>()(i) = counts[i];
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class ReadParquetOp : public OpKernel {
 public:
  explicit ReadParquetOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string& filename = filename_tensor.scalar<string>()();

    const Tensor& column_tensor = context->input(1);
    const string& column = column_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(2);
    const string& memory = memory_tensor.scalar<string>()();

    const Tensor& start_tensor = context->input(3);
    int64 start = start_tensor.scalar<int64>()();

    const Tensor& stop_tensor = context->input(4);
    int64 stop = stop_tensor.scalar<int64>()();

    std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory.data(), memory.size()));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    std::shared_ptr<ArrowRandomAccessFile> parquet_file(new ArrowRandomAccessFile(file.get(), size));
    std::unique_ptr<::parquet::ParquetFileReader> parquet_reader = parquet::ParquetFileReader::Open(parquet_file);
    std::shared_ptr<::parquet::FileMetaData> file_metadata = parquet_reader->metadata();
    int column_index = 0;
    while (column_index < file_metadata->num_columns()) {
      if (file_metadata->schema()->Column(column_index)->path().get()->ToDotString() == column) {
        break;
      }
      column_index++;
    }
    OP_REQUIRES(context, (column_index < file_metadata->num_columns()), errors::InvalidArgument("unable to find column: ", column));

    if (start > file_metadata->num_rows()) {
      start = file_metadata->num_rows();
    }
    if (stop < 0) {
        stop = file_metadata->num_rows();
    }
    if (stop > file_metadata->num_rows()) {
        stop = file_metadata->num_rows();
    }

    TensorShape output_shape({stop - start});

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    int64 row_group_offset = 0;
    for (int row_group = 0; row_group < file_metadata->num_row_groups(); row_group++) {
      std::shared_ptr<parquet::RowGroupReader> row_group_reader = parquet_reader->RowGroup(row_group);
      // Skip if row group is not within [start..stop]
      if ((row_group_offset + row_group_reader->metadata()->num_rows() < start) || (stop <= row_group_offset)) {
        row_group_offset += row_group_reader->metadata()->num_rows();
        continue;
      }
      // Find row_to_read range
      int64 row_to_read_start = row_group_offset > start ? row_group_offset : start;
      int64 row_to_read_final = (row_group_offset + row_group_reader->metadata()->num_rows()) < (stop) ? (row_group_offset + row_group_reader->metadata()->num_rows()) : (stop);
      int64 row_to_read_count = row_to_read_final - row_to_read_start;

      std::shared_ptr<parquet::ColumnReader> column_reader = row_group_reader->Column(column_index);

      // buffer to fill location is tensor.data()[row_to_read_start - start]

      #define PROCESS_TYPE(ptype, type) \
        { \
          parquet::TypedColumnReader<ptype>* reader = \
              static_cast<parquet::TypedColumnReader<ptype>*>( \
                  column_reader.get()); \
          if (row_to_read_start > row_group_offset) { \
            reader->Skip(row_to_read_start - row_group_offset); \
          } \
          ptype::c_type* value = (ptype::c_type *)(void *)(&(output_tensor->flat<type>().data()[row_to_read_start - start])); \
          int64_t values_read; \
          int64_t levels_read = reader->ReadBatch(row_to_read_count, nullptr, nullptr, value, &values_read); \
          OP_REQUIRES(context, (levels_read == values_read && levels_read == row_to_read_count), errors::InvalidArgument("null value in column: ", column)); \
        }
      switch (file_metadata->schema()->Column(column_index)->physical_type()) {
      case parquet::Type::BOOLEAN:
        PROCESS_TYPE(parquet::BooleanType, bool);
        break;
      case parquet::Type::INT32:
        PROCESS_TYPE(parquet::Int32Type, int32);
        break;
      case parquet::Type::INT64:
        PROCESS_TYPE(parquet::Int64Type, int64);
        break;
      case parquet::Type::FLOAT:
        PROCESS_TYPE(parquet::FloatType, float);
        break;
      case parquet::Type::DOUBLE:
        PROCESS_TYPE(parquet::DoubleType, double);
        break;
      default:
        OP_REQUIRES(context, false, errors::InvalidArgument("invalid data type: ", file_metadata->schema()->Column(column_index)->physical_type()));
      }
      row_group_offset += row_group_reader->metadata()->num_rows();
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("ListParquetColumns").Device(DEVICE_CPU),
                        ListParquetColumnsOp);
REGISTER_KERNEL_BUILDER(Name("ReadParquet").Device(DEVICE_CPU),
                        ReadParquetOp);


}  // namespace


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
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  std::shared_ptr<ArrowRandomAccessFile> parquet_file_;
  std::unique_ptr<::parquet::ParquetFileReader> parquet_reader_;
  std::shared_ptr<::parquet::FileMetaData> parquet_metadata_;

  std::vector<DataType> dtypes_;
  std::vector<TensorShape> shapes_;
  std::vector<string> columns_;
  std::unordered_map<string, int64> columns_index_;
};

REGISTER_KERNEL_BUILDER(Name("ParquetReadableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<ParquetReadable>);
REGISTER_KERNEL_BUILDER(Name("ParquetReadableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<ParquetReadable>);
REGISTER_KERNEL_BUILDER(Name("ParquetReadablePartitions").Device(DEVICE_CPU),
                        IOReadablePartitionsOp<ParquetReadable>);
REGISTER_KERNEL_BUILDER(Name("ParquetReadableRead").Device(DEVICE_CPU),
                        IOReadableReadOp<ParquetReadable>);
}  // namespace data
}  // namespace tensorflow
