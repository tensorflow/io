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
#include "tensorflow_io/arrow/kernels/arrow_kernels.h"
#include "tensorflow_io/core/kernels/io_interface.h"
#include "arrow/io/api.h"
#include "arrow/ipc/feather.h"
#include "arrow/ipc/feather_generated.h"
#include "arrow/buffer.h"
#include "arrow/adapters/tensorflow/convert.h"
#include "arrow/table.h"

namespace tensorflow {
namespace data {

Status GetTensorFlowType(std::shared_ptr<::arrow::DataType> dtype, ::tensorflow::DataType* out) {
  ::arrow::Status status = ::arrow::adapters::tensorflow::GetTensorFlowType(dtype, out);
  if (!status.ok()) {
    return errors::InvalidArgument("arrow data type ", dtype, " is not supported: ", status);
  }
  return Status::OK();
}

Status GetArrowType(::tensorflow::DataType dtype, std::shared_ptr<::arrow::DataType>* out) {
  ::arrow::Status status = ::arrow::adapters::tensorflow::GetArrowType(dtype, out);
  if (!status.ok()) {
    return errors::InvalidArgument("tensorflow data type ", dtype, " is not supported: ", status);
  }
  return Status::OK();
}

namespace {

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
      case ::arrow::ipc::feather::fbs::Type_BOOL:
        dtype = ::tensorflow::DataType::DT_BOOL;
        break;
      case ::arrow::ipc::feather::fbs::Type_INT8:
        dtype = ::tensorflow::DataType::DT_INT8;
        break;
      case ::arrow::ipc::feather::fbs::Type_INT16:
        dtype = ::tensorflow::DataType::DT_INT16;
        break;
      case ::arrow::ipc::feather::fbs::Type_INT32:
        dtype = ::tensorflow::DataType::DT_INT32;
        break;
      case ::arrow::ipc::feather::fbs::Type_INT64:
        dtype = ::tensorflow::DataType::DT_INT64;
        break;
      case ::arrow::ipc::feather::fbs::Type_UINT8:
        dtype = ::tensorflow::DataType::DT_UINT8;
        break;
      case ::arrow::ipc::feather::fbs::Type_UINT16:
        dtype = ::tensorflow::DataType::DT_UINT16;
        break;
      case ::arrow::ipc::feather::fbs::Type_UINT32:
        dtype = ::tensorflow::DataType::DT_UINT32;
        break;
      case ::arrow::ipc::feather::fbs::Type_UINT64:
        dtype = ::tensorflow::DataType::DT_UINT64;
        break;
      case ::arrow::ipc::feather::fbs::Type_FLOAT:
        dtype = ::tensorflow::DataType::DT_FLOAT;
        break;
      case ::arrow::ipc::feather::fbs::Type_DOUBLE:
        dtype = ::tensorflow::DataType::DT_DOUBLE;
        break;
      case ::arrow::ipc::feather::fbs::Type_UTF8:
      case ::arrow::ipc::feather::fbs::Type_BINARY:
      case ::arrow::ipc::feather::fbs::Type_CATEGORY:
      case ::arrow::ipc::feather::fbs::Type_TIMESTAMP:
      case ::arrow::ipc::feather::fbs::Type_DATE:
      case ::arrow::ipc::feather::fbs::Type_TIME:
      // case ::arrow::ipc::feather::fbs::Type_LARGE_UTF8:
      // case ::arrow::ipc::feather::fbs::Type_LARGE_BINARY:
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

REGISTER_KERNEL_BUILDER(Name("IoListFeatherColumns").Device(DEVICE_CPU),
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
      case ::arrow::ipc::feather::fbs::Type_BOOL:
        dtype = ::tensorflow::DataType::DT_BOOL;
        break;
      case ::arrow::ipc::feather::fbs::Type_INT8:
        dtype = ::tensorflow::DataType::DT_INT8;
        break;
      case ::arrow::ipc::feather::fbs::Type_INT16:
        dtype = ::tensorflow::DataType::DT_INT16;
        break;
      case ::arrow::ipc::feather::fbs::Type_INT32:
        dtype = ::tensorflow::DataType::DT_INT32;
        break;
      case ::arrow::ipc::feather::fbs::Type_INT64:
        dtype = ::tensorflow::DataType::DT_INT64;
        break;
      case ::arrow::ipc::feather::fbs::Type_UINT8:
        dtype = ::tensorflow::DataType::DT_UINT8;
        break;
      case ::arrow::ipc::feather::fbs::Type_UINT16:
        dtype = ::tensorflow::DataType::DT_UINT16;
        break;
      case ::arrow::ipc::feather::fbs::Type_UINT32:
        dtype = ::tensorflow::DataType::DT_UINT32;
        break;
      case ::arrow::ipc::feather::fbs::Type_UINT64:
        dtype = ::tensorflow::DataType::DT_UINT64;
        break;
      case ::arrow::ipc::feather::fbs::Type_FLOAT:
        dtype = ::tensorflow::DataType::DT_FLOAT;
        break;
      case ::arrow::ipc::feather::fbs::Type_DOUBLE:
        dtype = ::tensorflow::DataType::DT_DOUBLE;
        break;
      case ::arrow::ipc::feather::fbs::Type_UTF8:
      case ::arrow::ipc::feather::fbs::Type_BINARY:
      case ::arrow::ipc::feather::fbs::Type_CATEGORY:
      case ::arrow::ipc::feather::fbs::Type_TIMESTAMP:
      case ::arrow::ipc::feather::fbs::Type_DATE:
      case ::arrow::ipc::feather::fbs::Type_TIME:
      // case ::arrow::ipc::feather::fbs::Type_LARGE_UTF8:
      // case ::arrow::ipc::feather::fbs::Type_LARGE_BINARY:
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

    std::shared_ptr<arrow::Column> column;
    arrow::Status s = reader_->GetColumn(column_index, &column);
    if (!s.ok()) {
      return errors::Internal(s.ToString());
    }

    std::shared_ptr<::arrow::Column> slice = column->Slice(element_start, element_stop);

    #define FEATHER_PROCESS_TYPE(TTYPE,ATYPE) { \
        int64 curr_index = 0; \
        for (auto chunk : slice->data()->chunks()) { \
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

REGISTER_KERNEL_BUILDER(Name("IoFeatherReadableInit").Device(DEVICE_CPU),
                        IOInterfaceInitOp<FeatherReadable>);
REGISTER_KERNEL_BUILDER(Name("IoFeatherReadableSpec").Device(DEVICE_CPU),
                        IOInterfaceSpecOp<FeatherReadable>);
REGISTER_KERNEL_BUILDER(Name("IoFeatherReadableRead").Device(DEVICE_CPU),
                        IOReadableReadOp<FeatherReadable>);

}  // namespace data
}  // namespace tensorflow
