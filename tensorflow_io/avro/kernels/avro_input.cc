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
#include "api/DataFile.hh"
#include "api/Compiler.hh"
#include "api/Generic.hh"
#include "api/Stream.hh"
#include <sstream>

namespace tensorflow {
namespace data {
static const size_t kAvroDataInputStreamBufferSize = 8192;
class AvroDataInputStream : public avro::InputStream {
public:
  AvroDataInputStream(io::InputStreamInterface* s)
    : stream_(s) {}
  virtual ~AvroDataInputStream() {}
  bool next(const uint8_t** data, size_t* len) override {
    if (*len == 0) {
      *len = kAvroDataInputStreamBufferSize;
    }
    if (*len <= prefix_.size()) {
      buffer_ = prefix_.substr(0, *len);
      prefix_ = prefix_.substr(*len);
    } else {
      int64 bytes_to_read = *len - prefix_.size();
      string chunk;
      stream_->ReadNBytes(bytes_to_read, &chunk);
      buffer_ = std::move(prefix_);
      buffer_.append(chunk);
      prefix_.clear();
    }
    *data = (const uint8_t*)buffer_.data();
    *len =  buffer_.size();
    byte_count_ += *len;
    return (*len != 0);
  }
  void backup(size_t len) override {
    string chunk = buffer_.substr(buffer_.size() - len);
    chunk.append(prefix_);
    prefix_ = std::move(chunk);
    byte_count_ -= len;
  }
  void skip(size_t len) override {
    if (len <= prefix_.size()) {
      prefix_ = prefix_.substr(len);
    } else {
      int64 bytes_to_read = len - prefix_.size();
      stream_->SkipNBytes(bytes_to_read);
      prefix_.clear();
    }
    byte_count_ += len;
  }
  size_t byteCount() const override {
    return byte_count_;
  }
private:
  io::InputStreamInterface* stream_;
  size_t byte_count_ = 0;
  string prefix_;
  string buffer_;
};

class AvroInputStream{
public:
  explicit AvroInputStream(io::InputStreamInterface* s, const string& schema, const std::vector<string>& columns)
    : stream_(s)
    , schema_(schema)
    , columns_(columns)
    , reader_(nullptr) {
   }

  ~AvroInputStream() {
    reader_.reset(nullptr);
  }

  Status Open() {
    string error;
    std::istringstream ss(schema_);
    if (!avro::compileJsonSchema(ss, reader_schema_, error)) {
      return errors::Unimplemented("Avro schema error: ", error);
    }
    std::unique_ptr<avro::InputStream> stream(static_cast<avro::InputStream*>(new AvroDataInputStream(stream_)));
    reader_.reset(new avro::DataFileReader<avro::GenericDatum>(std::move(stream), reader_schema_));
    return Status::OK();
  }
  const avro::ValidSchema& ReaderSchema() const {
    return reader_schema_;
  }
  bool ReadDatum(avro::GenericDatum& datum) {
    return reader_->read(datum);
  }
private:
  io::InputStreamInterface* stream_;
  string schema_;
  std::vector<string> columns_;
  std::unique_ptr<avro::DataFileReader<avro::GenericDatum> > reader_;
  avro::ValidSchema reader_schema_;
};

class AvroInput: public FileInput<AvroInputStream> {
 public:
  Status ReadRecord(io::InputStreamInterface* s, IteratorContext* ctx, std::unique_ptr<AvroInputStream>& state, int64 record_to_read, int64* record_read, std::vector<Tensor>* out_tensors) const override {
    if (state.get() == nullptr) {
      state.reset(new AvroInputStream(s, schema(), columns()));
      TF_RETURN_IF_ERROR(state.get()->Open());
    }
    avro::GenericDatum datum(state.get()->ReaderSchema());
    while ((*record_read) < record_to_read && state.get()->ReadDatum(datum)) {
      const avro::GenericRecord& record = datum.value<avro::GenericRecord>();
      if (*record_read == 0) {
        out_tensors->clear();
        // Let's allocate enough space for Tensor, if more than read then slice.
        for (size_t i = 0; i < columns().size(); i++) {
          const string& column = columns()[i];
          const avro::GenericDatum& field = record.field(column);
          DataType dtype;
          switch (field.type()) {
          case avro::AVRO_BOOL:
            dtype = DT_BOOL;
            break;
          case avro::AVRO_INT:
            dtype = DT_INT32;
            break;
          case avro::AVRO_LONG:
            dtype = DT_INT64;
            break;
          case avro::AVRO_FLOAT:
            dtype = DT_FLOAT;
            break;
          case avro::AVRO_DOUBLE:
            dtype = DT_DOUBLE;
            break;
          case avro::AVRO_STRING:
            dtype = DT_STRING;
            break;
          case avro::AVRO_BYTES:
            dtype = DT_STRING;
            break;
          case avro::AVRO_FIXED:
            dtype = DT_STRING;
            break;
          case avro::AVRO_ENUM:
            dtype = DT_STRING;
            break;
          default:
            return errors::InvalidArgument("unsupported data type: ", field.type());
          }
          Tensor tensor(ctx->allocator({}), dtype, {record_to_read});
          out_tensors->emplace_back(std::move(tensor));
        }
      }
      for (size_t i = 0; i < columns().size(); i++) {
        const string& column = columns()[i];
        const avro::GenericDatum& field = record.field(column);
        switch (field.type()) {
        case avro::AVRO_BOOL:
          ((*out_tensors)[i]).flat<bool>()(*record_read) = field.value<bool>();
          break;
        case avro::AVRO_INT:
          ((*out_tensors)[i]).flat<int32>()(*record_read) = field.value<int32_t>();
          break;
        case avro::AVRO_LONG:
          ((*out_tensors)[i]).flat<int64>()(*record_read) = field.value<int64_t>();
          break;
        case avro::AVRO_FLOAT:
          ((*out_tensors)[i]).flat<float>()(*record_read) = field.value<float>();
          break;
        case avro::AVRO_DOUBLE:
          ((*out_tensors)[i]).flat<double>()(*record_read) = field.value<double>();
          break;
        case avro::AVRO_STRING:
          ((*out_tensors)[i]).flat<string>()(*record_read) = field.value<string>();
          break;
        case avro::AVRO_BYTES:
          {
            const std::vector<uint8_t>& value = field.value<std::vector<uint8_t>>();
            string v;
            if (value.size() > 0) {
              v.resize(value.size());
              memcpy(&v[0], &value[0], value.size());
            }
          }
          break;
        case avro::AVRO_FIXED:
          {
            const std::vector<uint8_t>& value = field.value<avro::GenericFixed>().value();
            string v;
            if (value.size() > 0) {
              v.resize(value.size());
              memcpy(&v[0], &value[0], value.size());
            }
          }
          break;
        case avro::AVRO_ENUM:
          ((*out_tensors)[i]).flat<string>()(*record_read) = field.value<avro::GenericEnum>().symbol();
          break;
        default:
          return errors::InvalidArgument("unsupported data type: ", field.type());
        }
      }
      (*record_read)++;
    }
    // Slice if needed
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

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(AvroInput, "tensorflow::data::AvroInput");

REGISTER_KERNEL_BUILDER(Name("AvroInput").Device(DEVICE_CPU),
                        FileInputOp<AvroInput>);
REGISTER_KERNEL_BUILDER(Name("AvroDataset").Device(DEVICE_CPU),
                        FileInputDatasetOp<AvroInput, AvroInputStream>);
}  // namespace data
}  // namespace tensorflow
