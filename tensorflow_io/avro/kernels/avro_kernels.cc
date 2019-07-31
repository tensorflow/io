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
#include "api/DataFile.hh"
#include "api/Compiler.hh"
#include "api/Generic.hh"
#include "api/Stream.hh"
#include "api/Validator.hh"

namespace tensorflow {
namespace data {
namespace {

static const size_t kAvroInputStreamBufferSize = 8192;
class AvroInputStream : public avro::SeekableInputStream {
public:
  AvroInputStream(tensorflow::RandomAccessFile* file)
    : file_(file) {
  }
  virtual ~AvroInputStream() {}
  bool next(const uint8_t** data, size_t* len) override {
    if (*len == 0) {
      *len = kAvroInputStreamBufferSize;
    }
    if (buffer_.size() < *len) {
      buffer_.resize(*len);
    }
    StringPiece result;
    Status status = file_->Read(byte_count_, *len, &result, &buffer_[0]);
    *data = (const uint8_t*)buffer_.data();
    *len =  result.size();
    byte_count_ += *len;
    return (*len != 0);
  }
  void backup(size_t len) override {
    byte_count_ -= len;
  }
  void skip(size_t len) override {
    byte_count_ += len;
  }
  void seek(int64_t position) override {
    byte_count_ = position;
  }
  size_t byteCount() const override {
    return byte_count_;
  }
private:
  tensorflow::RandomAccessFile* file_;
  string buffer_;
  uint64 byte_count_ = 0;
};

class ListAvroColumnsOp : public OpKernel {
 public:
  explicit ListAvroColumnsOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string filename = filename_tensor.scalar<string>()();

    const Tensor& schema_tensor = context->input(1);
    const string& schema = schema_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(2);
    const string& memory = memory_tensor.scalar<string>()();

    avro::ValidSchema reader_schema;

    string error;
    std::istringstream ss(schema);
    OP_REQUIRES(context, avro::compileJsonSchema(ss, reader_schema, error), errors::Unimplemented("Avro schema error: ", error));

    avro::GenericDatum datum(reader_schema.root());

    std::vector<string> columns;
    std::vector<string> dtypes;
    columns.reserve(reader_schema.root()->names());
    dtypes.reserve(reader_schema.root()->names());

    const avro::GenericRecord& record = datum.value<avro::GenericRecord>();
    for (int i = 0; i < reader_schema.root()->names(); i++) {
      const avro::GenericDatum& field = record.field(reader_schema.root()->nameAt(i));
      string dtype;
      switch(field.type()) {
      case avro::AVRO_BOOL:
        dtype = "bool";
        break;
      case avro::AVRO_INT:
        dtype = "int32";
        break;
      case avro::AVRO_LONG:
        dtype = "int64";
        break;
      case avro::AVRO_FLOAT:
        dtype = "float";
        break;
      case avro::AVRO_DOUBLE:
        dtype = "double";
        break;
      case avro::AVRO_STRING:
        dtype = "string";
        break;
      case avro::AVRO_BYTES:
        dtype = "string";
        break;
      case avro::AVRO_FIXED:
        dtype = "string";
        break;
      case avro::AVRO_ENUM:
        dtype = "string";
        break;
      default:
        break;
      }
      if (dtype == "") {
        continue;
      }
      columns.emplace_back(reader_schema.root()->nameAt(i));
      dtypes.emplace_back(dtype);
    }

    TensorShape output_shape = filename_tensor.shape();
    output_shape.AddDim(columns.size());

    Tensor* columns_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &columns_tensor));
    Tensor* dtypes_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &dtypes_tensor));

    output_shape.AddDim(1);

    for (size_t i = 0; i < columns.size(); i++) {
      columns_tensor->flat<string>()(i) = columns[i];
      dtypes_tensor->flat<string>()(i) = dtypes[i];
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class ReadAvroOp : public OpKernel {
 public:
  explicit ReadAvroOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filename_tensor = context->input(0);
    const string& filename = filename_tensor.scalar<string>()();

    const Tensor& schema_tensor = context->input(1);
    const string& schema = schema_tensor.scalar<string>()();

    const Tensor& column_tensor = context->input(2);
    const string& column = column_tensor.scalar<string>()();

    const Tensor& memory_tensor = context->input(3);
    const string& memory = memory_tensor.scalar<string>()();

    const Tensor& offset_tensor = context->input(4);
    const int64 offset = offset_tensor.scalar<int64>()();

    const Tensor& length_tensor = context->input(5);
    int64 length = length_tensor.scalar<int64>()();

    avro::ValidSchema reader_schema;

    string error;
    std::istringstream ss(schema);
    OP_REQUIRES(context, avro::compileJsonSchema(ss, reader_schema, error), errors::Unimplemented("Avro schema error: ", error));


    std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory.data(), memory.size()));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    if (length < 0) {
      length = size - offset;
    }

    avro::GenericDatum datum(reader_schema);

    std::unique_ptr<avro::InputStream> stream(new AvroInputStream(file.get()));
    std::unique_ptr<avro::DataFileReader<avro::GenericDatum>> reader(new avro::DataFileReader<avro::GenericDatum>(std::move(stream), reader_schema));

    if (offset != 0) {
      reader->sync(offset);
    }

    #define BOOL_VALUE records.push_back(field.value<bool>())
    #define INT32_VALUE records.emplace_back(field.value<int32_t>())
    #define INT64_VALUE records.emplace_back(field.value<int64_t>())
    #define FLOAT_VALUE records.emplace_back(field.value<float>())
    #define DOUBLE_VALUE records.emplace_back(field.value<double>())
    #define STRING_VALUE records.emplace_back(field.value<string>())
    #define BYTES_VALUE { \
            const std::vector<uint8_t>& value = field.value<std::vector<uint8_t>>(); \
            string v; \
            if (value.size() > 0) { \
              v.resize(value.size()); \
              memcpy(&v[0], &value[0], value.size()); \
            } \
            records.emplace_back(v); \
          }
    #define FIXED_VALUE { \
            const std::vector<uint8_t>& value = field.value<avro::GenericFixed>().value(); \
            string v; \
            if (value.size() > 0) { \
              v.resize(value.size()); \
              memcpy(&v[0], &value[0], value.size()); \
            } \
	    records.emplace_back(v); \
          }
    #define ENUM_VALUE records.emplace_back(field.value<avro::GenericEnum>().symbol())

    #define PROCESS_RECORD(TYPE, ATYPE, VALUE) { \
      std::vector<TYPE> records; \
      while (!reader->pastSync(offset + length) && reader->read(datum)) { \
        const avro::GenericRecord& record = datum.value<avro::GenericRecord>(); \
        const avro::GenericDatum& field = record.field(column); \
        VALUE; \
      } \
      Tensor* output_tensor; \
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({static_cast<int64>(records.size())}), &output_tensor)); \
      for (size_t i = 0; i < records.size(); i++) { \
        output_tensor->flat<TYPE>()(i) = std::move(records[i]); \
      } \
    }
    switch(datum.value<avro::GenericRecord>().field(column).type()) {
    case avro::AVRO_BOOL:
      PROCESS_RECORD(bool, bool, BOOL_VALUE);
      break;
    case avro::AVRO_INT:
      PROCESS_RECORD(int32, int32_t, INT32_VALUE);
      break;
    case avro::AVRO_LONG:
      PROCESS_RECORD(int64, int64_t, INT64_VALUE);
      break;
    case avro::AVRO_FLOAT:
      PROCESS_RECORD(float, float, FLOAT_VALUE);
      break;
    case avro::AVRO_DOUBLE:
      PROCESS_RECORD(double, double, DOUBLE_VALUE);
      break;
    case avro::AVRO_STRING:
      PROCESS_RECORD(string, string, STRING_VALUE);
      break;
    case avro::AVRO_BYTES:
      PROCESS_RECORD(string, string, BYTES_VALUE);
      break;
    case avro::AVRO_FIXED:
      PROCESS_RECORD(string, string, FIXED_VALUE);
      break;
    case avro::AVRO_ENUM:
      PROCESS_RECORD(string, string, ENUM_VALUE);
      break;
    default:
      OP_REQUIRES(context, false, errors::InvalidArgument("unsupported data type: ", datum.value<avro::GenericRecord>().field(column).type()));
    }

  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("ListAvroColumns").Device(DEVICE_CPU),
                        ListAvroColumnsOp);
REGISTER_KERNEL_BUILDER(Name("ReadAvro").Device(DEVICE_CPU),
                        ReadAvroOp);


}  // namespace
}  // namespace data
}  // namespace tensorflow
