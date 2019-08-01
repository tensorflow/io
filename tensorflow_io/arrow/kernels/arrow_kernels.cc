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
#include "arrow/io/api.h"
#include "arrow/ipc/feather.h"
#include "arrow/table.h"

namespace arrow {
namespace adapters {
namespace tensorflow {
Status GetTensorFlowType(std::shared_ptr<DataType> dtype, ::tensorflow::DataType* out);
}
}
}
namespace tensorflow {
namespace data {
namespace {

// NOTE: Both SizedRandomAccessFile and ArrowRandomAccessFile overlap
// with another PR. Will remove duplicate once PR merged

// Note: This SizedRandomAccessFile should only lives within Compute()
// of the kernel as buffer could be released by outside.
class SizedRandomAccessFile : public tensorflow::RandomAccessFile {
 public:
  SizedRandomAccessFile(Env* env, const string& filename, const string& optional_memory)
  : file_(nullptr)
  , size_status_(Status::OK())
  , size_(optional_memory.size())
  , buffer_(optional_memory) {
    if (size_ == 0) {
      size_status_ = env->GetFileSize(filename, &size_);
      if (size_status_.ok()) {
        size_status_ = env->NewRandomAccessFile(filename, &file_);
      }
    }
  }

  virtual ~SizedRandomAccessFile() {}
  Status Read(uint64 offset, size_t n, StringPiece* result, char* scratch) const override {
    if (file_.get() != nullptr) {
      return file_.get()->Read(offset, n, result, scratch);
    }
    size_t bytes_to_read = 0;
    if (offset < size_) {
      bytes_to_read = (offset + n < size_) ? n : (size_ - offset);
    }
    if (bytes_to_read > 0) {
      memcpy(scratch, &buffer_.data()[offset], bytes_to_read);
    }
    *result = StringPiece(scratch, bytes_to_read);
    if (bytes_to_read < n) {
      return errors::OutOfRange("EOF reached");
    }
    return Status::OK();
  }
  Status GetFileSize(uint64* size) {
    if (size_status_.ok()) {
      *size = size_;
    }
    return size_status_;
  }
 private:
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
  Status size_status_;
  uint64 size_;
  const string& buffer_;
};

class ArrowRandomAccessFile : public ::arrow::io::RandomAccessFile {
public:
  explicit ArrowRandomAccessFile(tensorflow::RandomAccessFile *file, int64 size)
    : file_(file)
    , size_(size) { }

  ~ArrowRandomAccessFile() {}
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

class ListFeatherColumnsOp : public OpKernel {
 public:
  explicit ListFeatherColumnsOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string filename = filename_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(1);
    const string& memory = memory_tensor.scalar<string>()();
    std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    std::shared_ptr<ArrowRandomAccessFile> feather_file(new ArrowRandomAccessFile(file.get(), size));

    std::unique_ptr<arrow::ipc::feather::TableReader> reader;
    arrow::Status s = arrow::ipc::feather::TableReader::Open(feather_file, &reader);
    OP_REQUIRES(context, s.ok(), errors::Internal(s.ToString()));

    std::vector<string> columns;
    std::vector<string> dtypes;
    std::vector<int64> counts;
    columns.reserve(reader->num_columns());
    dtypes.reserve(reader->num_columns());
    counts.reserve(reader->num_columns());
    for (int i = 0; i < reader->num_columns(); i++) {
      std::shared_ptr<arrow::Column> column;
      s = reader->GetColumn(i, &column);
      OP_REQUIRES(context, s.ok(), errors::Internal(s.ToString()));

      ::tensorflow::DataType data_type;
      s = ::arrow::adapters::tensorflow::GetTensorFlowType(column->type(), &data_type);
      if (!s.ok()) {
        continue;
      }
      string dtype = "";
      switch (data_type) {
      case ::tensorflow::DT_BOOL:
        dtype = "bool";
        break;
      case ::tensorflow::DT_UINT8:
        dtype = "uint8";
        break;
      case ::tensorflow::DT_INT8:
        dtype = "int8";
        break;
      case ::tensorflow::DT_UINT16:
        dtype = "uint16";
        break;
      case ::tensorflow::DT_INT16:
        dtype = "int16";
        break;
      case ::tensorflow::DT_UINT32:
        dtype = "uint32";
        break;
      case ::tensorflow::DT_INT32:
        dtype = "int32";
        break;
      case ::tensorflow::DT_UINT64:
        dtype = "uint64";
        break;
      case ::tensorflow::DT_INT64:
        dtype = "int64";
        break;
      case ::tensorflow::DT_HALF:
        dtype = "half";
        break;
      case ::tensorflow::DT_FLOAT:
        dtype = "float";
        break;
      case ::tensorflow::DT_DOUBLE:
        dtype = "double";
        break;
      default:
       break; 
      }
      if (dtype == "") {
        continue;
      }
      columns.push_back(reader->GetColumnName(i));
      dtypes.push_back(dtype);
      counts.push_back(reader->num_rows());
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

REGISTER_KERNEL_BUILDER(Name("ListFeatherColumns").Device(DEVICE_CPU),
                        ListFeatherColumnsOp);


}  // namespace
}  // namespace data
}  // namespace tensorflow
