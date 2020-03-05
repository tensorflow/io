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

#include "arrow/io/api.h"
#include "arrow/ipc/api.h"
#include "arrow/ipc/feather.h"
#include "generated/feather_generated.h"
#include "arrow/array.h"
#include "arrow/buffer.h"
#include "arrow/table.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_io/arrow/kernels/arrow_kernels.h"
#include "tensorflow_io/arrow/kernels/arrow_util.h"
#include "tensorflow_io/core/kernels/io_interface.h"

namespace tensorflow {
namespace data {

namespace {

class ArrowReadableResourceBase : public ResourceBase {
 public:
  virtual Status Init(const std::shared_ptr<arrow::Table>& table) = 0;
  virtual Status Spec(int32 column_index,
                      PartialTensorShape* shape,
                      DataType* dtype) = 0;
  virtual Status Read(int64 start,
                      int64 stop,
                      int32 column_index,
                      Tensor* value) = 0;
};

class ArrowReadableResource : public ArrowReadableResourceBase {
 public:
  ArrowReadableResource(Env* env) : env_(env) {}
  ~ArrowReadableResource() {}

  Status Init(const std::shared_ptr<arrow::Table>& table) override {
    mutex_lock l(mu_);
    table_ = table;
    return Status::OK();
  }

  Status Spec(int32 column_index, PartialTensorShape* shape, DataType* dtype) override {
    mutex_lock l(mu_);
    if (column_index < 0 || column_index >= table_->num_columns()) {
      return errors::InvalidArgument("Invalid column index: ", column_index);
    }

    // Get shape from first array chunk and replace outer dim with row count
    auto chunked_arr = table_->column(column_index);
    auto arr = chunked_arr->chunk(0);
    TensorShape tmp_shape = TensorShape({});

    TF_RETURN_IF_ERROR(
        ArrowUtil::AssignSpec(arr, 0, arr->length(), dtype, &tmp_shape));

    gtl::InlinedVector<int64, 4> dims = tmp_shape.dim_sizes();
    dims[0] = table_->num_rows();
    *shape = TensorShape(dims);

    return Status::OK();
  }

  Status Read(int64 start, int64 stop, int32 column_index, Tensor* value) override {
    mutex_lock l(mu_);
    if (column_index < 0 || column_index >= table_->num_columns()) {
      return errors::InvalidArgument("Invalid column index: ", column_index);
    }

    auto chunked_arr = table_->column(column_index);

    // Slice with requested start/stop index
    if (start > 0 || stop - start < table_->num_rows()) {
      chunked_arr = chunked_arr->Slice(start, stop - start);
      start = 0;  // Slice will offset chunked_arr from start position
    }

    // Column is empty
    if (chunked_arr->num_chunks() == 0) {
      return Status::OK();
    }
    // Convert the array
    else if (chunked_arr->num_chunks() == 1) {
      auto arr = chunked_arr->chunk(0);
      TF_RETURN_IF_ERROR(ArrowUtil::AssignTensor(arr, start, value));
    }
    // Convert each chunk at a time
    else {
      int64 length_converted = 0;
      for (int i = 0; i < chunked_arr->num_chunks(); ++i) {
        auto arr = chunked_arr->chunk(i);

        // Take a slice that will share the underlying TensorBuffer
        auto slice = value->Slice(length_converted, length_converted + arr->length());

        TF_RETURN_IF_ERROR(ArrowUtil::AssignTensor(arr, start, &slice));
        length_converted += arr->length();
      }
    }

    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return "ArrowReadableResource";
  }

 protected:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::shared_ptr<arrow::Table> table_ GUARDED_BY(mu_);
};

class ArrowReadableFromMemoryInitOp : public ResourceOpKernel<ArrowReadableResource> {
 public:
  explicit ArrowReadableFromMemoryInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<ArrowReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<ArrowReadableResource>::Compute(context);

    const Tensor* schema_buffer_addr_tensor;
    OP_REQUIRES_OK(context, context->input("schema_buffer_address", &schema_buffer_addr_tensor));
    uint64 schema_buffer_addr = schema_buffer_addr_tensor->scalar<uint64>()();
    const uint8_t* schema_buffer = reinterpret_cast<const uint8_t*>(schema_buffer_addr);

    const Tensor* schema_buffer_size_tensor;
    OP_REQUIRES_OK(context, context->input("schema_buffer_size", &schema_buffer_size_tensor));
    int64 schema_buffer_size = schema_buffer_size_tensor->scalar<int64>()();

    auto buffer_ = std::make_shared<arrow::Buffer>(
        schema_buffer,
        schema_buffer_size);
    auto buffer_reader = std::make_shared<arrow::io::BufferReader>(buffer_);

    std::shared_ptr<arrow::Schema> schema;
    arrow::Status status = arrow::ipc::ReadSchema(buffer_reader.get(), nullptr, &schema);
    OP_REQUIRES(context, status.ok(), errors::Internal("Error reading Arrow Schema"));

    const Tensor* array_buffer_addrs_tensor;
    OP_REQUIRES_OK(context, context->input("array_buffer_addresses", &array_buffer_addrs_tensor));
    const TensorShape& array_addrs_shape = array_buffer_addrs_tensor->shape();
    OP_REQUIRES(context, array_addrs_shape.dims() == 3,
        errors::InvalidArgument("array_buffer_addresses should be 3 dimensional"));

    const Tensor* array_buffer_sizes_tensor;
    OP_REQUIRES_OK(context, context->input("array_buffer_sizes", &array_buffer_sizes_tensor));
    const TensorShape& array_sizes_shape = array_buffer_sizes_tensor->shape();
    OP_REQUIRES(context, array_addrs_shape == array_sizes_shape,
        errors::InvalidArgument("array_buffer_sizes should be 3 dimensional"));

    const Tensor* array_lengths_tensor;
    OP_REQUIRES_OK(context, context->input("array_lengths", &array_lengths_tensor));
    OP_REQUIRES(context, array_lengths_tensor->shape().dims() == 3,
        errors::InvalidArgument("array_lengths should be 3 dimensional"));

    auto array_addrs = array_buffer_addrs_tensor->tensor<uint64, 3>();
    auto array_sizes = array_buffer_sizes_tensor->tensor<int64, 3>();
    auto array_lengths = array_lengths_tensor->tensor<int64, 3>();

    int64 num_columns = array_addrs_shape.dim_size(0);
    int64 num_chunks = array_addrs_shape.dim_size(1);
    int64 num_buffers = array_addrs_shape.dim_size(2);

    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
    columns.reserve(num_columns);

    for (int64 i = 0; i < num_columns; ++i) {
      auto field = schema->field(i);
      arrow::ArrayVector chunks;
      chunks.reserve(num_chunks);

      for (int64 j = 0; j < num_chunks; ++j) {
        std::vector<std::shared_ptr<arrow::Buffer>> buffers;
        buffers.reserve(num_buffers);

        for (int64 k = 0; k < num_buffers; ++k) {
          uint64 array_ij_addr = array_addrs(i, j, k);
          int64 array_ij_size = array_sizes(i, j, k);

          // If size is < 0, then don't add a buffer, if size == 0 then add empty buffer
          if (array_ij_size > 0) {
            const uint8_t* array_ij_buffer = reinterpret_cast<const uint8_t*>(array_ij_addr);
            buffers.push_back(std::make_shared<arrow::Buffer>(array_ij_buffer, array_ij_size));
          } else if (array_ij_size == 0) {
            buffers.push_back(std::shared_ptr<arrow::Buffer>());
          }
        }

        // Make the Array chunk, TODO null values not currently supported
        std::vector<int64> lengths;
        for (int64 k = 0; k < array_lengths_tensor->shape().dim_size(2); ++k) {
          int64 length = array_lengths(i, j, k);
          if (length > 0) {
            lengths.push_back(length);
          }
        }
        std::shared_ptr<arrow::ArrayData> array_data;
        OP_REQUIRES_OK(context,
            ArrowUtil::MakeArrayData(field->type(), lengths, buffers, &array_data));
        auto array = arrow::MakeArray(array_data);
        chunks.push_back(array);
      }

      auto chunked_array = std::make_shared<arrow::ChunkedArray>(chunks);
      columns.push_back(chunked_array);
    }

    auto table = arrow::Table::Make(schema, columns);
    OP_REQUIRES_OK(context, resource_->Init(table));
  }

  Status CreateResource(ArrowReadableResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new ArrowReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class ArrowReadableSpecOp : public OpKernel {
 public:
  explicit ArrowReadableSpecOp(OpKernelConstruction* context)
      : OpKernel(context), column_index_(-1) {
    int32 column_index;
    Status status = context->GetAttr("column_index", &column_index);
    if (status.ok()) {
      column_index_ = column_index;
    }
  }

  void Compute(OpKernelContext* context) override {
    ArrowReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    PartialTensorShape shape;
    DataType dtype;
    OP_REQUIRES_OK(context, resource->Spec(column_index_, &shape, &dtype));

    Tensor shape_tensor(DT_INT64, TensorShape({shape.dims()}));
    for (int64 i = 0; i < shape.dims(); i++) {
      shape_tensor.flat<int64>()(i) = shape.dim_size(i);
    }
    Tensor dtype_tensor(DT_INT64, TensorShape({}));
    dtype_tensor.scalar<int64>()() = dtype;
    context->set_output(0, shape_tensor);
    context->set_output(1, dtype_tensor);
  }

 private:
  int32 column_index_;
};

class ArrowReadableReadOp : public OpKernel {
 public:
  explicit ArrowReadableReadOp(OpKernelConstruction* context)
      : OpKernel(context), column_index_(-1) {
    int32 column_index;
    Status status = context->GetAttr("column_index", &column_index);
    if (status.ok()) {
      column_index_ = column_index;
    }
  }

  void Compute(OpKernelContext* context) override {
    ArrowReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    int64 start = start_tensor->scalar<int64>()();

    const Tensor* stop_tensor;
    OP_REQUIRES_OK(context, context->input("stop", &stop_tensor));
    int64 stop = stop_tensor->scalar<int64>()();

    // Get the spec from the table
    PartialTensorShape shape;
    DataType dtype;
    OP_REQUIRES_OK(context, resource->Spec(column_index_, &shape, &dtype));

    // Verify requested records
    int64 length = shape.dim_size(0);
    OP_REQUIRES(context, start >= 0 && start < length && stop > start,
                errors::InvalidArgument("Invalid start, stop inputs: ", start, ", ", stop));

    // Cap the stop record at the length
    if (stop > length) {
      stop = length;
    }

    // Create shape of the output tensor
    gtl::InlinedVector<int64, 4> dims = shape.dim_sizes();
    dims[0] = stop - start;
    TensorShape value_shape(dims);

    // Allocate the output tensor
    Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, value_shape, &value_tensor));

    // Read the output tensor value
    OP_REQUIRES_OK(context, resource->Read(start, stop, column_index_, value_tensor));
  }

 private:
  int32 column_index_;
};

REGISTER_KERNEL_BUILDER(Name("IO>ArrowReadableFromMemoryInit").Device(DEVICE_CPU),
                        ArrowReadableFromMemoryInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>ArrowReadableSpec").Device(DEVICE_CPU),
                        ArrowReadableSpecOp);
REGISTER_KERNEL_BUILDER(Name("IO>ArrowReadableRead").Device(DEVICE_CPU),
                        ArrowReadableReadOp);


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
    std::unique_ptr<SizedRandomAccessFile> file(new SizedRandomAccessFile(env_, filename, memory.data(), memory.size()));
    uint64 size;
    OP_REQUIRES_OK(context, file->GetFileSize(&size));

    // FEA1.....[metadata][uint32 metadata_length]FEA1
    static constexpr const char* kFeatherMagicBytes = "FEA1";

    size_t header_length = strlen(kFeatherMagicBytes);
    size_t footer_length = sizeof(uint32) + strlen(kFeatherMagicBytes);

    string buffer;
    buffer.resize(header_length > footer_length ? header_length : footer_length);

    StringPiece result;

    OP_REQUIRES_OK(context, file->Read(0, header_length, &result, &buffer[0]));
    OP_REQUIRES(context, !memcmp(buffer.data(), kFeatherMagicBytes, header_length), errors::InvalidArgument("not a feather file"));

    OP_REQUIRES_OK(context, file->Read(size - footer_length, footer_length, &result, &buffer[0]));
    OP_REQUIRES(context, !memcmp(buffer.data() + sizeof(uint32), kFeatherMagicBytes, footer_length - sizeof(uint32)), errors::InvalidArgument("incomplete feather file"));

    uint32 metadata_length = *reinterpret_cast<const uint32*>(buffer.data());

    buffer.resize(metadata_length);

    OP_REQUIRES_OK(context, file->Read(size - footer_length - metadata_length, metadata_length, &result, &buffer[0]));

    const ::arrow::ipc::feather::fbs::CTable* table = ::arrow::ipc::feather::fbs::GetCTable(buffer.data());

    OP_REQUIRES(context, (table->version() >= ::arrow::ipc::feather::kFeatherVersion), errors::InvalidArgument("feather file is old: ", table->version(), " vs. ", ::arrow::ipc::feather::kFeatherVersion));

    std::vector<string> columns;
    std::vector<string> dtypes;
    std::vector<int64> counts;
    columns.reserve(table->columns()->size());
    dtypes.reserve(table->columns()->size());
    counts.reserve(table->columns()->size());

    for (int64 i = 0; i < table->columns()->size(); i++) {
      DataType dtype = ::tensorflow::DataType::DT_INVALID;
      switch (table->columns()->Get(i)->values()->type()) {
      case ::arrow::ipc::feather::fbs::Type::BOOL:
        dtype = ::tensorflow::DataType::DT_BOOL;
        break;
      case ::arrow::ipc::feather::fbs::Type::INT8:
        dtype = ::tensorflow::DataType::DT_INT8;
        break;
      case ::arrow::ipc::feather::fbs::Type::INT16:
        dtype = ::tensorflow::DataType::DT_INT16;
        break;
      case ::arrow::ipc::feather::fbs::Type::INT32:
        dtype = ::tensorflow::DataType::DT_INT32;
        break;
      case ::arrow::ipc::feather::fbs::Type::INT64:
        dtype = ::tensorflow::DataType::DT_INT64;
        break;
      case ::arrow::ipc::feather::fbs::Type::UINT8:
        dtype = ::tensorflow::DataType::DT_UINT8;
        break;
      case ::arrow::ipc::feather::fbs::Type::UINT16:
        dtype = ::tensorflow::DataType::DT_UINT16;
        break;
      case ::arrow::ipc::feather::fbs::Type::UINT32:
        dtype = ::tensorflow::DataType::DT_UINT32;
        break;
      case ::arrow::ipc::feather::fbs::Type::UINT64:
        dtype = ::tensorflow::DataType::DT_UINT64;
        break;
      case ::arrow::ipc::feather::fbs::Type::FLOAT:
        dtype = ::tensorflow::DataType::DT_FLOAT;
        break;
      case ::arrow::ipc::feather::fbs::Type::DOUBLE:
        dtype = ::tensorflow::DataType::DT_DOUBLE;
        break;
      case ::arrow::ipc::feather::fbs::Type::UTF8:
      case ::arrow::ipc::feather::fbs::Type::BINARY:
      case ::arrow::ipc::feather::fbs::Type::CATEGORY:
      case ::arrow::ipc::feather::fbs::Type::TIMESTAMP:
      case ::arrow::ipc::feather::fbs::Type::DATE:
      case ::arrow::ipc::feather::fbs::Type::TIME:
      // case ::arrow::ipc::feather::fbs::Type::LARGE_UTF8:
      // case ::arrow::ipc::feather::fbs::Type::LARGE_BINARY:
      default:
        break;
      }
      columns.push_back(table->columns()->Get(i)->name()->str());
      dtypes.push_back(::tensorflow::DataTypeString(dtype));
      counts.push_back(table->num_rows());
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

    for (size_t i = 0; i < columns.size(); i++) {
      columns_tensor->flat<string>()(i) = columns[i];
      dtypes_tensor->flat<string>()(i) = dtypes[i];
      shapes_tensor->flat<int64>()(i) = counts[i];
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>ListFeatherColumns").Device(DEVICE_CPU),
                        ListFeatherColumnsOp);


}  // namespace


class FeatherReadable : public IOReadableInterface {
 public:
  FeatherReadable(Env* env)
  : env_(env) {}

  ~FeatherReadable() {}
  Status Init(const std::vector<string>& input, const std::vector<string>& metadata, const void* memory_data, const int64 memory_size) override {
    if (input.size() > 1) {
      return errors::InvalidArgument("more than 1 filename is not supported");
    }

    const string& filename = input[0];
    file_.reset(new SizedRandomAccessFile(env_, filename, memory_data, memory_size));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    // FEA1.....[metadata][uint32 metadata_length]FEA1
    static constexpr const char* kFeatherMagicBytes = "FEA1";

    size_t header_length = strlen(kFeatherMagicBytes);
    size_t footer_length = sizeof(uint32) + strlen(kFeatherMagicBytes);

    string buffer;
    buffer.resize(header_length > footer_length ? header_length : footer_length);

    StringPiece result;

    TF_RETURN_IF_ERROR(file_->Read(0, header_length, &result, &buffer[0]));
    if (memcmp(buffer.data(), kFeatherMagicBytes, header_length) != 0) {
      return errors::InvalidArgument("not a feather file");
    }

    TF_RETURN_IF_ERROR(file_->Read(file_size_ - footer_length, footer_length, &result, &buffer[0]));
    if (memcmp(buffer.data() + sizeof(uint32), kFeatherMagicBytes, footer_length - sizeof(uint32)) != 0) {
      return errors::InvalidArgument("incomplete feather file");
    }

    uint32 metadata_length = *reinterpret_cast<const uint32*>(buffer.data());

    buffer.resize(metadata_length);

    TF_RETURN_IF_ERROR(file_->Read(file_size_ - footer_length - metadata_length, metadata_length, &result, &buffer[0]));

    const ::arrow::ipc::feather::fbs::CTable* table = ::arrow::ipc::feather::fbs::GetCTable(buffer.data());

    if (table->version() < ::arrow::ipc::feather::kFeatherVersion) {
      return errors::InvalidArgument("feather file is old: ", table->version(), " vs. ", ::arrow::ipc::feather::kFeatherVersion);
    }

    for (int i = 0; i < table->columns()->size(); i++) {
      ::tensorflow::DataType dtype = ::tensorflow::DataType::DT_INVALID;
      switch (table->columns()->Get(i)->values()->type()) {
      case ::arrow::ipc::feather::fbs::Type::BOOL:
        dtype = ::tensorflow::DataType::DT_BOOL;
        break;
      case ::arrow::ipc::feather::fbs::Type::INT8:
        dtype = ::tensorflow::DataType::DT_INT8;
        break;
      case ::arrow::ipc::feather::fbs::Type::INT16:
        dtype = ::tensorflow::DataType::DT_INT16;
        break;
      case ::arrow::ipc::feather::fbs::Type::INT32:
        dtype = ::tensorflow::DataType::DT_INT32;
        break;
      case ::arrow::ipc::feather::fbs::Type::INT64:
        dtype = ::tensorflow::DataType::DT_INT64;
        break;
      case ::arrow::ipc::feather::fbs::Type::UINT8:
        dtype = ::tensorflow::DataType::DT_UINT8;
        break;
      case ::arrow::ipc::feather::fbs::Type::UINT16:
        dtype = ::tensorflow::DataType::DT_UINT16;
        break;
      case ::arrow::ipc::feather::fbs::Type::UINT32:
        dtype = ::tensorflow::DataType::DT_UINT32;
        break;
      case ::arrow::ipc::feather::fbs::Type::UINT64:
        dtype = ::tensorflow::DataType::DT_UINT64;
        break;
      case ::arrow::ipc::feather::fbs::Type::FLOAT:
        dtype = ::tensorflow::DataType::DT_FLOAT;
        break;
      case ::arrow::ipc::feather::fbs::Type::DOUBLE:
        dtype = ::tensorflow::DataType::DT_DOUBLE;
        break;
      case ::arrow::ipc::feather::fbs::Type::UTF8:
      case ::arrow::ipc::feather::fbs::Type::BINARY:
      case ::arrow::ipc::feather::fbs::Type::CATEGORY:
      case ::arrow::ipc::feather::fbs::Type::TIMESTAMP:
      case ::arrow::ipc::feather::fbs::Type::DATE:
      case ::arrow::ipc::feather::fbs::Type::TIME:
      // case ::arrow::ipc::feather::fbs::Type::LARGE_UTF8:
      // case ::arrow::ipc::feather::fbs::Type::LARGE_BINARY:
      default:
        break;
      }
      shapes_.push_back(TensorShape({static_cast<int64>(table->num_rows())}));
      dtypes_.push_back(dtype);
      columns_.push_back(table->columns()->Get(i)->name()->str());
      columns_index_[table->columns()->Get(i)->name()->str()] = i;
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
    int64 element_start = start < shapes_[column_index].dim_size(0) ? start : shapes_[column_index].dim_size(0);
    int64 element_stop = stop < shapes_[column_index].dim_size(0) ? stop : shapes_[column_index].dim_size(0);

    if (element_start > element_stop) {
      return errors::InvalidArgument("dataset selection is out of boundary");
    }
    if (element_start == element_stop) {
      return Status::OK();
    }

    if (feather_file_.get() == nullptr) {
      feather_file_.reset(new ArrowRandomAccessFile(file_.get(), file_size_));
      arrow::Status s = arrow::ipc::feather::TableReader::Open(feather_file_, &reader_);
      if (!s.ok()) {
        return errors::Internal(s.ToString());
      }
    }

    std::shared_ptr<arrow::ChunkedArray> column;
    arrow::Status s = reader_->GetColumn(column_index, &column);
    if (!s.ok()) {
      return errors::Internal(s.ToString());
    }

    std::shared_ptr<::arrow::ChunkedArray> slice = column->Slice(element_start, element_stop);

    #define FEATHER_PROCESS_TYPE(TTYPE,ATYPE) { \
        int64 curr_index = 0; \
        for (auto chunk : slice->chunks()) { \
         for (int64_t item = 0; item < chunk->length(); item++) { \
            value->flat<TTYPE>()(curr_index) = (dynamic_cast<ATYPE *>(chunk.get()))->Value(item); \
            curr_index++; \
          } \
        } \
      }
    switch (value->dtype()) {
    case DT_BOOL:
      FEATHER_PROCESS_TYPE(bool, ::arrow::BooleanArray);
      break;
    case DT_INT8:
      FEATHER_PROCESS_TYPE(int8, ::arrow::NumericArray<::arrow::Int8Type>);
      break;
    case DT_UINT8:
      FEATHER_PROCESS_TYPE(uint8, ::arrow::NumericArray<::arrow::UInt8Type>);
      break;
    case DT_INT16:
      FEATHER_PROCESS_TYPE(int16, ::arrow::NumericArray<::arrow::Int16Type>);
      break;
    case DT_UINT16:
      FEATHER_PROCESS_TYPE(uint16, ::arrow::NumericArray<::arrow::UInt16Type>);
      break;
    case DT_INT32:
      FEATHER_PROCESS_TYPE(int32, ::arrow::NumericArray<::arrow::Int32Type>);
      break;
    case DT_UINT32:
      FEATHER_PROCESS_TYPE(uint32, ::arrow::NumericArray<::arrow::UInt32Type>);
      break;
    case DT_INT64:
      FEATHER_PROCESS_TYPE(int64, ::arrow::NumericArray<::arrow::Int64Type>);
      break;
    case DT_UINT64:
      FEATHER_PROCESS_TYPE(uint64, ::arrow::NumericArray<::arrow::UInt64Type>);
      break;
    case DT_FLOAT:
      FEATHER_PROCESS_TYPE(float, ::arrow::NumericArray<::arrow::FloatType>);
      break;
    case DT_DOUBLE:
      FEATHER_PROCESS_TYPE(double, ::arrow::NumericArray<::arrow::DoubleType>);
      break;
    default:
      return errors::InvalidArgument("data type is not supported: ", DataTypeString(value->dtype()));
    }
    (*record_read) = element_stop - element_start;
    return Status::OK();
  }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("FeatherReadable");
  }
 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ GUARDED_BY(mu_);
  uint64 file_size_ GUARDED_BY(mu_);
  std::shared_ptr<ArrowRandomAccessFile> feather_file_ GUARDED_BY(mu_);
  std::unique_ptr<arrow::ipc::feather::TableReader> reader_ GUARDED_BY(mu_);

  std::vector<DataType> dtypes_;
  std::vector<TensorShape> shapes_;
  std::vector<string> columns_;
  std::unordered_map<string, int64> columns_index_;
};

REGISTER_KERNEL_BUILDER(Name("IO>FeatherReadableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<FeatherReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>FeatherReadableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<FeatherReadable>);
REGISTER_KERNEL_BUILDER(Name("IO>FeatherReadableRead").Device(DEVICE_CPU),
                        IOReadableReadOp<FeatherReadable>);

}  // namespace data
}  // namespace tensorflow
