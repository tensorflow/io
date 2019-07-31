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
#include "tensorflow/core/lib/io/buffered_inputstream.h"

namespace tensorflow {
namespace data {
namespace {

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
      memcpy(scratch, buffer_.data(), bytes_to_read);
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
class FilenoInputStream : public io::InputStreamInterface {
 public:
  FilenoInputStream(int fileno) : fileno_(fileno) {}
  virtual ~FilenoInputStream() {}

  virtual Status ReadNBytes(int64 bytes_to_read, string* result) override {
    if (bytes_to_read < 0) {
      return errors::InvalidArgument("Can't read a negative number of bytes: ", bytes_to_read);
    }

    result->clear();
    if (final_) {
      return errors::OutOfRange("EOF reached");
    }


    string buffer;
    result->resize(bytes_to_read);
    int64 bytes_read = 0;
    while (bytes_read <  bytes_to_read) {
      size_t chunk = bytes_to_read - bytes_read;
      ssize_t returned = read(fileno_, &(*result)[bytes_read], chunk);
      if (returned < 0) {
        result->resize(bytes_read);
        return errors::Internal("read fileno ", fileno_, " error: ", returned);
      }
      if (returned == 0) {
        break;
      }
      bytes_read += returned;
    }
    offset_ += bytes_read;
    result->resize(bytes_read);
    if (bytes_read < bytes_to_read) {
      return errors::OutOfRange("EOF reached");
    }
    return Status::OK();
  }

  virtual int64 Tell() const override {
    return offset_;
  }

  virtual Status Reset() override {
    return errors::Unimplemented("Reset fileno stream is not implemented");
  }
 private:
  int fileno_ = -1;
  int64 offset_ = 0;
  bool final_ = false;
};

class ReadTextOp : public OpKernel {
 public:
  explicit ReadTextOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string& filename = filename_tensor.scalar<string>()();

    const Tensor& offset_tensor = context->input(1);
    const int64 offset = offset_tensor.scalar<int64>()();

    const Tensor& length_tensor = context->input(2);
    int64 length = length_tensor.scalar<int64>()();

    const Tensor& memory_tensor = context->input(3);
    const string& memory = memory_tensor.scalar<string>()();

    std::vector<string> lines;

    if (filename == "file://-" || filename == "file://0") {
      // If we read from stdin then let's read until EOF is reached
      // Note: It is possible to read data in large slices. However,
      // BufferedInputStream takes a cached buffer which complicates
      // the data read from stream. Will need to implement a no-cache
      // version of ReadLine() in order to read chunks.
      std::unique_ptr<FilenoInputStream> input_stream(new FilenoInputStream(STDIN_FILENO));
      std::unique_ptr<tensorflow::io::BufferedInputStream> stream(new tensorflow::io::BufferedInputStream(input_stream.get(), 4096));

      Status status = Status::OK();
      while (status.ok()) {
        string line;
        status = stream->ReadLine(&line);
        OP_REQUIRES(context, (status.ok() || errors::IsOutOfRange(status)), status);
        if (!status.ok()) {
          break;
        }
        lines.emplace_back(line);
      }
    } else {
      std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory));
      uint64 size;
      OP_REQUIRES_OK(context, file->GetFileSize(&size));
      if (length < 0) {
        length = size;
      }

      // This ReadText is a splittable version so that it is possible to read Text from a chunk of a file,
      // much like Hadoop. We use the following method to decide if a line belongs to the chunk or not:
      // 1) offset = 0: read lines and stop after length is reached.
      // 2) offset > 0: back off 1 and skip one line to start with the next line, stop after length is reached.
      //
      // Note: We use BufferedInputStream which is only able to process separator of "\n", though it could
      // be expanded to more than "\n" in the future.

      std::unique_ptr<tensorflow::io::BufferedInputStream> stream(new tensorflow::io::BufferedInputStream(file.get(), 65536));
      if (offset > 0) {
        OP_REQUIRES_OK(context, stream->SkipNBytes(offset - 1));
        string line;
        OP_REQUIRES_OK(context, stream->ReadLine(&line));
      }

      while (stream->Tell() < offset + length) {
        string line;
        Status status = stream->ReadLine(&line);
        OP_REQUIRES(context, (status.ok() || errors::IsOutOfRange(status)), status);
        if (!status.ok()) {
          break;
        }
        lines.emplace_back(line);
      }
    }

    TensorShape output_shape({static_cast<int64>(lines.size())});

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    for (size_t i = 0; i < lines.size(); i++) {
      output_tensor->flat<string>()(i) = std::move(lines[i]);
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("ReadText").Device(DEVICE_CPU),
                        ReadTextOp);


}  // namespace
}  // namespace data
}  // namespace tensorflow
