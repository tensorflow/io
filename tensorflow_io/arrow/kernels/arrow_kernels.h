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

#include "kernels/stream.h"
#include "arrow/io/api.h"
#include "arrow/buffer.h"

namespace tensorflow {
namespace data {

// NOTE: Both SizedRandomAccessFile and ArrowRandomAccessFile overlap
// with another PR. Will remove duplicate once PR merged

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
}  // namespace data
}  // namespace tensorflow
