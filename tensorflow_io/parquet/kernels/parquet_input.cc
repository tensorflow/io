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

#include "kernels/dataset_ops.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "parquet/api/reader.h"

namespace tensorflow {
namespace data {

class SizedRandomAccessBufferedStream : public SizedRandomAccessInputStreamInterface {
public:
  explicit SizedRandomAccessBufferedStream(io::InputStreamInterface* s)
    : input_stream_(s) { }
  Status GetFileSize(uint64* file_size) override {
    // TODO: This is not necessary the best format as it needs
    // two pass to get the buffer. Could be enhanced later.
    if (size_ >= 0) {
      *file_size = size_;
      return Status::OK();
    }
    std::vector<string> buffer;
    do {
      string chunk;
      Status status = input_stream_->ReadNBytes(4096, &chunk);
      if (!(status.ok() || errors::IsOutOfRange(status))) {
        return status;
      }
      if (chunk.size() > 0) {
        buffer.emplace_back(std::move(chunk));
      }
      if (!status.ok()) {
        break;
      }
    } while (true);
    size_ = 0;
    for (size_t i = 0; i < buffer.size(); i++) {
        size_ += buffer[i].size();
    }
    buffer_.clear();
    buffer_.reserve(size_);
    for (size_t i = 0; i < buffer.size(); i++) {
        buffer_.append(buffer[i]);
    }
    buffer.clear();

    *file_size = size_;
    return  Status::OK();
  }
  Status Read(uint64 offset, size_t n, StringPiece* result, char* scratch) const override {
    Status status = Status::OK();
    if (offset + n > size_) {
      status = errors::OutOfRange("EOF reached: ", result->size(), " bytes read, ", n, " requested");
      n = size_ - offset;
    }
    memcpy(scratch, &buffer_.data()[offset], n);
    *result = StringPiece(scratch, n);
    return status;
  }
  Status ReadNBytes(int64 bytes_to_read, string* result) override {
    return input_stream_->ReadNBytes(bytes_to_read, result);
  }
  int64 Tell() const override {
    return input_stream_->Tell();
  }
  Status Reset() override {
    return input_stream_->Reset();
  }
private:
  io::InputStreamInterface* input_stream_;
  string buffer_;
  int64 size_ = -1;
};

class ParquetRandomAccessFile : public ::arrow::io::RandomAccessFile {
public:
  explicit ParquetRandomAccessFile(io::InputStreamInterface* s)
    : input_stream_(nullptr)
    , buffered_stream_(nullptr) {
    input_stream_ = dynamic_cast<SizedRandomAccessInputStreamInterface*>(s);
    if (input_stream_ == nullptr) {
      buffered_stream_.reset(new SizedRandomAccessBufferedStream(s));
      input_stream_ = buffered_stream_.get();
    }
  }
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
    uint64 size_value = 0;
    Status status = input_stream_->GetFileSize(&size_value);
    if (!status.ok()) {
      return arrow::Status::IOError(status.error_message());
    }
    *size = size_value;
    return arrow::Status::OK();
  }
  bool supports_zero_copy() const override {
    return false;
  }
  arrow::Status ReadAt(int64_t position, int64_t nbytes, int64_t* bytes_read, void* out) override {
    StringPiece result;
    Status status = input_stream_->Read(position, nbytes, &result, (char *)out);
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
    Status status = input_stream_->Read(position, nbytes, &result, &buffer[0]);
    if (!(status.ok() || errors::IsOutOfRange(status))) {
        return arrow::Status::IOError(status.error_message());
    }
    buffer.resize(result.size());
    return arrow::Buffer::FromString(buffer, out);
  }
private:
  SizedRandomAccessInputStreamInterface* input_stream_;
  std::unique_ptr<SizedRandomAccessBufferedStream> buffered_stream_;
};

class ParquetInputStream{
public:
  explicit ParquetInputStream(io::InputStreamInterface* s, const std::vector<string>& columns)
    : input_stream_(new ParquetRandomAccessFile(s))
    , column_names_(columns) {
  }
  Status ReadHeader() {
    parquet_reader_ = parquet::ParquetFileReader::Open(input_stream_);
    file_metadata_ = parquet_reader_->metadata();
    columns_ = std::vector<int64>(column_names_.size(), -1);
    dtypes_ = std::vector<DataType>(column_names_.size());
    for (size_t i = 0; i < column_names_.size(); i++) {
      for (int j = 0; j < file_metadata_->schema()->num_columns(); j++) {
        if (column_names_[i] == file_metadata_->schema()->Column(j)->path().get()->ToDotString()) {
          columns_[i] = j;
          switch(file_metadata_->schema()->Column(j)->physical_type()) {
          case parquet::Type::BOOLEAN:
            dtypes_[i] = DT_BOOL;
            break;
          case parquet::Type::INT32:
            dtypes_[i] = DT_INT32;
            break;
          case parquet::Type::INT64:
            dtypes_[i] = DT_INT64;
            break;
          case parquet::Type::FLOAT:
            dtypes_[i] = DT_FLOAT;
            break;
          case parquet::Type::DOUBLE:
            dtypes_[i] = DT_DOUBLE;
            break;
          default:
            return errors::InvalidArgument("data type is not supported for column ", column_names_[i]);
          }
          break;
        }
      }
      if (columns_[i] < 0) {
        return errors::InvalidArgument("unable to find column ", column_names_[i]);
      }
    }
    current_row_group_ = 0;
    TF_RETURN_IF_ERROR(ReadRowGroup());
    return Status::OK();
  }
  DataType DType(int64 i) {
    return dtypes_[i];
  }
  int64 Columns() {
    return (int64)columns_.size();
  }
  Status ReadRowGroup() {
    if (current_row_group_ < file_metadata_->num_row_groups()) {
      row_group_reader_ = parquet_reader_->RowGroup(current_row_group_);
      column_readers_.clear();
      for (size_t i = 0; i < columns_.size(); i++) {
        int64 column = columns_[i];
        std::shared_ptr<parquet::ColumnReader> column_reader =
            row_group_reader_->Column(column);
        column_readers_.emplace_back(column_reader);
      }
    }
    current_row_ = 0;
    return Status::OK();
  }
  ~ParquetInputStream() {
    current_row_ = 0;
    column_readers_.clear();
    row_group_reader_.reset();
    current_row_group_ = 0;
    file_metadata_.reset();
    parquet_reader_.reset();
  }
  Status ReadRecord(int64 index, int64 record_to_read, std::vector<Tensor>* out_tensors, int64* record_read) {
    while (current_row_group_ < file_metadata_->num_row_groups()) {
      if (current_row_ < row_group_reader_->metadata()->num_rows()) {
        // Read columns to outputs.
        // TODO: Read more than one value at a time.
        for (size_t i = 0; i < columns_.size(); i++) {
          DataType dtype = dtypes_[i];
          std::shared_ptr<parquet::ColumnReader> column_reader = column_readers_[i];
          TF_RETURN_IF_ERROR(GetTensorValue(current_row_, dtype, column_reader.get(), &(*out_tensors)[i], index));
        }
        ++current_row_;
        *record_read = 1;
        return Status::OK();
      }
      // We have reached the end of the current row group, so maybe
      // move on to next row group.
      current_row_ = 0;
      row_group_reader_.reset();
      ++current_row_group_;
      TF_RETURN_IF_ERROR(ReadRowGroup());
    }
    return Status::OK();
  }
private:
  template <typename DType>
  Status FillTensorValue(parquet::ColumnReader* column_reader,
                         typename DType::c_type* value) {
    parquet::TypedColumnReader<DType>* reader =
        static_cast<parquet::TypedColumnReader<DType>*>(column_reader);
    // Read one value at a time. The number of rows read is returned.
    // values_read contains the number of non-null rows
    int64_t values_read = 0;
    int64_t rows_read = reader->ReadBatch(1, nullptr, nullptr, value, &values_read);
    // Ensure only one value is read and there are no NULL values in the
    // rows read
    if (rows_read != 1) {
      return errors::Internal("rows_read (", rows_read, ") != 1 or values_read (", values_read, ") != 1");
    }
    return Status::OK();
  }
  Status GetTensorValue(int64 row, const DataType& data_type, parquet::ColumnReader* column_reader, Tensor* tensor, int64 index) {
    switch (data_type) {
      case DT_INT32: {
        parquet::TypedColumnReader<parquet::Int32Type>* reader =
            static_cast<parquet::TypedColumnReader<parquet::Int32Type>*>(
                column_reader);
        int32_t value;
        TF_RETURN_IF_ERROR(
            FillTensorValue<parquet::Int32Type>(reader, &value));
        tensor->flat<int32>()(index) = value;
      } break;
      case DT_INT64: {
        parquet::TypedColumnReader<parquet::Int64Type>* reader =
            static_cast<parquet::TypedColumnReader<parquet::Int64Type>*>(
                column_reader);
        int64_t value;
        TF_RETURN_IF_ERROR(
            FillTensorValue<parquet::Int64Type>(reader, &value));
        tensor->flat<int64>()(index) = value;
      } break;
      case DT_FLOAT: {
        parquet::TypedColumnReader<parquet::FloatType>* reader =
            static_cast<parquet::TypedColumnReader<parquet::FloatType>*>(
                column_reader);
        float value;
        TF_RETURN_IF_ERROR(
            FillTensorValue<parquet::FloatType>(reader, &value));
        tensor->flat<float>()(index) = value;
      } break;
      case DT_DOUBLE: {
        parquet::TypedColumnReader<parquet::DoubleType>* reader =
            static_cast<parquet::TypedColumnReader<parquet::DoubleType>*>(
                column_reader);
        double value;
        TF_RETURN_IF_ERROR(
            FillTensorValue<parquet::DoubleType>(reader, &value));
        tensor->flat<double>()(index) = value;
      } break;
      case DT_BOOL: {
        parquet::TypedColumnReader<parquet::BooleanType>* reader =
            static_cast<parquet::TypedColumnReader<parquet::BooleanType>*>(
                column_reader);
        bool value;
        TF_RETURN_IF_ERROR(
            FillTensorValue<parquet::BooleanType>(reader, &value));
        tensor->flat<bool>()(index) = value;
      } break;
      default:
        return errors::Unimplemented(
            DataTypeString(data_type),
            " is currently not supported in ParquetDataset");
    }
    return Status::OK();
  }
  std::shared_ptr<::arrow::io::RandomAccessFile> input_stream_;
  std::vector<string> column_names_;
  std::vector<int64> columns_;
  std::vector<DataType> dtypes_;
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader_;
  std::shared_ptr<parquet::FileMetaData> file_metadata_;
  int64 current_row_group_ = 0;
  std::shared_ptr<parquet::RowGroupReader> row_group_reader_;
  std::vector<std::shared_ptr<parquet::ColumnReader>> column_readers_;
  int64 current_row_ = 0;
};

class ParquetInput: public FileInput<ParquetInputStream> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<ParquetInputStream>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new ParquetInputStream(s, columns()));
      TF_RETURN_IF_ERROR(state.get()->ReadHeader());
    }
    // Let's allocate enough space for Tensor, if more than read, replace.
    for (int64 i = 0; i < state.get()->Columns(); i++) {
      Tensor tensor(ctx->allocator({}), state.get()->DType(i), {record_to_read});
      out_tensors->emplace_back(std::move(tensor));
    }
    while ((*record_read) < record_to_read) {
      int64 count = 0;
      TF_RETURN_IF_ERROR(state.get()->ReadRecord((*record_read), record_to_read - (*record_read), out_tensors, &count));
      (*record_read) += count;
      if (count == 0) {
        break;
      }
    }
    if (*record_read < record_to_read) {
      if (*record_read == 0) {
        out_tensors->clear();
      }
      for (size_t i = 0; i < out_tensors->size(); i++) {
        Tensor tensor = (*out_tensors)[i].Slice(0, *record_read);
        (*out_tensors)[i] = std::move(tensor);
      }
    }
    return Status::OK();
  }
  Status FromStream(io::InputStreamInterface* s) override {
    return Status::OK();
  }
  void EncodeAttributes(VariantTensorData* data) const override {
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    return true;
  }
 protected:
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(ParquetInput, "tensorflow::data::ParquetInput");

REGISTER_KERNEL_BUILDER(Name("ParquetInput").Device(DEVICE_CPU),
                        FileInputOp<ParquetInput>);
REGISTER_KERNEL_BUILDER(Name("ParquetDataset").Device(DEVICE_CPU),
                        FileInputDatasetOp<ParquetInput, ParquetInputStream>);
}  // namespace data
}  // namespace tensorflow
