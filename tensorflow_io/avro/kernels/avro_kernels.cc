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
#include "tensorflow_io/core/kernels/io_stream.h"
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

REGISTER_KERNEL_BUILDER(Name("IO>ListAvroColumns").Device(DEVICE_CPU),
                        ListAvroColumnsOp);
REGISTER_KERNEL_BUILDER(Name("IO>ReadAvro").Device(DEVICE_CPU),
                        ReadAvroOp);



}  // namespace

class AvroReadable : public IOReadableInterface {
 public:
  AvroReadable(Env* env)
  : env_(env) {}

  ~AvroReadable() {}
  Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) override {
    if (input.size() > 1) {
      return errors::InvalidArgument("more than 1 filename is not supported");
    }
    const string& filename = input[0];
    file_.reset(new SizedRandomAccessFile(env_, filename, memory_data, memory_size));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    string schema;
    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find("schema: ") == 0) {
        schema = metadata[i].substr(8);
      }
    }

    string error;
    std::istringstream ss(schema);
    if (!(avro::compileJsonSchema(ss, reader_schema_, error))) {
      return errors::Internal("Avro schema error: ", error);
    }

    for (int i = 0; i < reader_schema_.root()->names(); i++) {
      columns_.push_back(reader_schema_.root()->nameAt(i));
      columns_index_[reader_schema_.root()->nameAt(i)] = i;
    }

    avro::GenericDatum datum(reader_schema_.root());
    const avro::GenericRecord& record = datum.value<avro::GenericRecord>();
    for (size_t i = 0; i < reader_schema_.root()->names(); i++) {
      const avro::GenericDatum& field = record.field(columns_[i]);
      ::tensorflow::DataType dtype;
      switch(field.type()) {
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
        return errors::InvalidArgument("Avro type unsupported: ", field.type());
      }
      dtypes_.emplace_back(dtype);
    }

    // Find out the total number of rows
    reader_stream_.reset(new AvroInputStream(file_.get()));
    reader_.reset(new avro::DataFileReader<avro::GenericDatum>(std::move(reader_stream_), reader_schema_));

    avro::DecoderPtr decoder = avro::binaryDecoder();

    int64 total = 0;

    reader_->sync(0);
    int64 offset = reader_->previousSync();
    while (offset < file_size_) {
      StringPiece result;
      string buffer(16, 0x00);
      TF_RETURN_IF_ERROR(file_->Read(offset, buffer.size(), &result, &buffer[0]));
      std::unique_ptr<avro::InputStream> in = avro::memoryInputStream((const uint8_t*)result.data(), result.size());
      decoder->init(*in);
      long items = decoder->decodeLong();

      total += static_cast<int64>(items);
      positions_.emplace_back(std::pair<int64, int64>(static_cast<int64>(items), offset));

      reader_->sync(offset);
      offset = reader_->previousSync();
    }

    for (size_t i = 0; i < columns_.size(); i++) {
      shapes_.emplace_back(TensorShape({total}));
    }
    return Status::OK();
  }

  Status Partitions(std::vector<int64> *partitions) override {
    partitions->clear();
    // positions_ are pairs of <items, offset>
    for (size_t i = 0; i < positions_.size(); i++) {
      partitions->emplace_back(positions_[i].first);
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

    avro::GenericDatum datum(reader_schema_);

    // Find the start sync point
    int64 item_index_sync = 0;
    for (size_t i = 0; i < positions_.size(); i++, item_index_sync += positions_[i].first) {
      if (item_index_sync >= element_stop) {
        continue;
      }
      if (item_index_sync + positions_[i].first <= element_start) {
        continue;
      }
      // TODO: Avro is sync point partitioned and each block is very similiar to
      // Row Group of parquet. Ideally each block should be cached with the hope
      // that slicing and indexing will happend around the same block across multiple
      // rows. Caching is not done yet.

      // Seek to sync
      reader_->seek(positions_[i].second);
      for (int64 item_index = item_index_sync; item_index < (item_index_sync + positions_[i].first) && item_index < element_stop; item_index++) {
        // Read anyway
        if (!reader_->read(datum)) {
          return errors::Internal("unable to read record at: ", item_index);
        }
        // Assign only when in range
        if (item_index >= element_start) {
          const avro::GenericRecord& record = datum.value<avro::GenericRecord>();
          const avro::GenericDatum& field = record.field(column);
          switch(field.type()) {
          case avro::AVRO_BOOL:
            value->flat<bool>()(item_index - element_start) = field.value<bool>();
            break;
          case avro::AVRO_INT:
            value->flat<int32>()(item_index - element_start) = field.value<int32_t>();
            break;
          case avro::AVRO_LONG:
            value->flat<int64>()(item_index - element_start) = field.value<int64_t>();
            break;
          case avro::AVRO_FLOAT:
            value->flat<float>()(item_index - element_start) = field.value<float>();
            break;
          case avro::AVRO_DOUBLE:
            value->flat<double>()(item_index - element_start) = field.value<double>();
            break;
          case avro::AVRO_STRING:
            value->flat<string>()(item_index - element_start) = field.value<string>();
            break;
          case avro::AVRO_BYTES: {
              const std::vector<uint8_t>& field_value = field.value<std::vector<uint8_t>>();
              value->flat<string>()(item_index - element_start) = string((char *)&field_value[0], field_value.size());
            }
            break;
          case avro::AVRO_FIXED: {
              const std::vector<uint8_t>& field_value = field.value<avro::GenericFixed>().value();
              value->flat<string>()(item_index - element_start) = string((char *)&field_value[0], field_value.size());
            }
            break;
          case avro::AVRO_ENUM:
            value->flat<string>()(item_index - element_start) = field.value<avro::GenericEnum>().symbol();
            break;
          default:
            return errors::InvalidArgument("unsupported data type: ", field.type());
          }
        }
      }
    }
    (*record_read) = element_stop - element_start;
    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("AvroReadable");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  avro::ValidSchema reader_schema_;
  std::unique_ptr<avro::InputStream> reader_stream_;
  std::unique_ptr<avro::DataFileReader<avro::GenericDatum>> reader_;
  std::vector<std::pair<int64, int64>> positions_; // <items/sync> pair

  std::vector<DataType> dtypes_;
  std::vector<TensorShape> shapes_;
  std::vector<string> columns_;
  std::unordered_map<string, int64> columns_index_;
};

REGISTER_KERNEL_BUILDER(Name("IO>AvroReadableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<AvroReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>AvroReadableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<AvroReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>AvroReadablePartitions").Device(DEVICE_CPU),
                        IOReadablePartitionsOp<AvroReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>AvroReadableRead").Device(DEVICE_CPU),
                        IOReadableReadOp<AvroReadable>);

}  // namespace data
}  // namespace tensorflow
