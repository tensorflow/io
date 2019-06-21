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

namespace tensorflow {
namespace data {

class TextInput: public FileInput<io::BufferedInputStream> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<io::BufferedInputStream>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new io::BufferedInputStream(s, 4096));
    }
    Tensor value_tensor(ctx->allocator({}), DT_STRING, {record_to_read});
    while ((*record_read) < record_to_read) {
      string buffer;
      buffer.clear();
      Status status = state.get()->ReadLine(&buffer);
      if (!(status.ok() || errors::IsOutOfRange(status))) {
        return status;
      }
      if (!status.ok()) {
        break;
      }
      value_tensor.flat<string>()((*record_read)) = std::move(buffer);
      (*record_read)++;
    }
    if (*record_read > 0) {
      out_tensors->emplace_back(std::move(value_tensor));
    }
    return Status::OK();
  }
  Status FromStream(io::InputStreamInterface* s) override {
    // TODO: Read 4K buffer to detect BOM.
    //string header;
    //TF_RETURN_IF_ERROR(s.ReadNBytes(4096, &header));
    //for (size i = 0; i < header.size(); i++) {
    //  if (!isprint(header[i])) {
    //    return errors::InvalidArgument("text file contains character that is non printable at ", i);
    //  }
    //}
    return Status::OK();
  }
  void EncodeAttributes(VariantTensorData* data) const override {
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    return true;
  }
 protected:
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(TextInput, "tensorflow::data::TextInput");

REGISTER_KERNEL_BUILDER(Name("TextInput").Device(DEVICE_CPU),
                        FileInputOp<TextInput>);
REGISTER_KERNEL_BUILDER(Name("TextDataset").Device(DEVICE_CPU),
                        FileInputDatasetOp<TextInput, io::BufferedInputStream>);



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

class TextStreamInput: public StreamInput<io::BufferedInputStream> {
 public:
  Status ReadRecord(IteratorContext* ctx, std::unique_ptr<io::BufferedInputStream>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      if (endpoint() != "file://-") {
        return errors::InvalidArgument("only file://- (stdin) is supported for stream: ", endpoint());
      }
      state.reset(new io::BufferedInputStream(new FilenoInputStream(STDIN_FILENO), 4096, true));
    }
    Tensor value_tensor(ctx->allocator({}), DT_STRING, {record_to_read});
    while ((*record_read) < record_to_read) {
      string buffer;
      buffer.clear();
      Status status = state.get()->ReadLine(&buffer);
      if (!(status.ok() || errors::IsOutOfRange(status))) {
        return status;
      }
      if (!status.ok()) {
        break;
      }
      value_tensor.flat<string>()((*record_read)) = std::move(buffer);
      (*record_read)++;
    }
    if (*record_read > 0) {
      out_tensors->emplace_back(std::move(value_tensor));
    }
    return Status::OK();
  }
  Status FromEndpoint(const string& endpoint) override {
    if (endpoint != "file://-") {
      return errors::InvalidArgument("only file://- (stdin) is supported for stream: ", endpoint);
    }
std::cerr << "ENDPOINT: " << endpoint << std::endl;
    return Status::OK();
  }
  void EncodeAttributes(VariantTensorData* data) const override {
  }
  bool DecodeAttributes(const VariantTensorData& data) override {
    return true;
  }
 protected:
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(TextStreamInput, "tensorflow::data::TextStreamInput");

REGISTER_KERNEL_BUILDER(Name("TextStreamInput").Device(DEVICE_CPU),
                        StreamInputOp<TextStreamInput>);
REGISTER_KERNEL_BUILDER(Name("TextStreamDataset").Device(DEVICE_CPU),
                        StreamInputDatasetOp<TextStreamInput, io::BufferedInputStream>);
}  // namespace data
}  // namespace tensorflow
