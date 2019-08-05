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
#include "tensorflow_io/core/kernels/stream.h"
#include "parquet/api/reader.h"

namespace tensorflow {
namespace data {
namespace {

class ParquetRandomAccessFile : public ::arrow::io::RandomAccessFile {
public:
  explicit ParquetRandomAccessFile(tensorflow::RandomAccessFile *file, int64 size)
    : file_(file)
    , size_(size) { }

  ~ParquetRandomAccessFile() {}
  arrow::Status Close() override {
    return arrow::Status::OK();
  }
  arrow::Status Tell(int64_t* position) const override {
    return arrow::Status::NotImplemented("Tell");
  }
  arrow::Status Seek(int64_t position) override {
    return arrow::Status::NotImplemented("Seek");
  }
  arrow::Status Read(int64_t nbytes, int64_t* bytes_read, void* out) override {
    return arrow::Status::NotImplemented("Read (void*)");
  }
  arrow::Status Read(int64_t nbytes, std::shared_ptr<arrow::Buffer>* out) override {
    return arrow::Status::NotImplemented("Read (Buffer*)");
  }
  arrow::Status GetSize(int64_t* size) override {
    *size = size_;
    return arrow::Status::OK();
  }
  bool supports_zero_copy() const override {
    return false;
  }
  arrow::Status ReadAt(int64_t position, int64_t nbytes, int64_t* bytes_read, void* out) override {
    StringPiece result;
    Status status = file_->Read(position, nbytes, &result, (char*)out);
    if (!(status.ok() || errors::IsOutOfRange(status))) {
        return arrow::Status::IOError(status.error_message());
    }
    *bytes_read = result.size();
    return arrow::Status::OK();
  }
  arrow::Status ReadAt(int64_t position, int64_t nbytes, std::shared_ptr<arrow::Buffer>* out) override {
    string buffer;
    buffer.resize(nbytes);
    StringPiece result;
    Status status = file_->Read(position, nbytes, &result, (char*)(&buffer[0]));
    if (!(status.ok() || errors::IsOutOfRange(status))) {
        return arrow::Status::IOError(status.error_message());
    }
    buffer.resize(result.size());
    return arrow::Buffer::FromString(buffer, out);
  }
private:
  tensorflow::RandomAccessFile* file_;
  int64 size_;
};

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

    std::shared_ptr<ParquetRandomAccessFile> parquet_file(new ParquetRandomAccessFile(file.get(), size));
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

    std::shared_ptr<ParquetRandomAccessFile> parquet_file(new ParquetRandomAccessFile(file.get(), size));
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
}  // namespace data
}  // namespace tensorflow
