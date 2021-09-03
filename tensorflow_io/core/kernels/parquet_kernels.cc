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

#include "parquet/api/reader.h"
#include "parquet/windows_compatibility.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow_io/core/kernels/arrow/arrow_kernels.h"
#include "tensorflow_io/core/kernels/io_kernel.h"

namespace tensorflow {
namespace data {
namespace {

class ParquetReadableResource : public ResourceBase {
 public:
  ParquetReadableResource(Env* env) : env_(env) {}

  virtual ~ParquetReadableResource() {}

  Status Init(const string& input) {
    mutex_lock l(mu_);
    Status status = env_->IsDirectory(input);
    if (status.ok()) {
      return errors::InvalidArgument(
          "passing a directory path to 'filename' is not supported. ",
          "Use 'tf.data.Dataset.list_files()' with a map() operation instead.");
    }

    file_.reset(new SizedRandomAccessFile(env_, input, nullptr, 0));
    TF_RETURN_IF_ERROR(file_->GetFileSize(&file_size_));

    parquet_file_.reset(new ArrowRandomAccessFile(file_.get(), file_size_));

    parquet_file_.reset(new ArrowRandomAccessFile(file_.get(), file_size_));
    parquet_reader_ = parquet::ParquetFileReader::Open(parquet_file_);
    parquet_metadata_ = parquet_reader_->metadata();

    shapes_.clear();
    dtypes_.clear();
    columns_.clear();
    for (size_t i = 0; i < parquet_metadata_->num_columns(); i++) {
      ::tensorflow::DataType dtype;
      switch (parquet_metadata_->schema()->Column(i)->physical_type()) {
        case parquet::Type::BOOLEAN:
          dtype = ::tensorflow::DT_BOOL;
          break;
        case parquet::Type::INT32:
          dtype = ::tensorflow::DT_INT32;
          break;
        case parquet::Type::INT64:
          dtype = ::tensorflow::DT_INT64;
          break;
        case parquet::Type::INT96:  // Deprecated, thrown out exception when
                                    // access with __getitem__
          dtype = ::tensorflow::DT_INT64;
          break;
        case parquet::Type::FLOAT:
          dtype = ::tensorflow::DT_FLOAT;
          break;
        case parquet::Type::DOUBLE:
          dtype = ::tensorflow::DT_DOUBLE;
          break;
        case parquet::Type::BYTE_ARRAY:
          dtype = ::tensorflow::DT_STRING;
          break;
        case parquet::Type::FIXED_LEN_BYTE_ARRAY:
          dtype = ::tensorflow::DT_STRING;
          break;
        default:
          return errors::InvalidArgument(
              "parquet data type is not supported: ",
              parquet_metadata_->schema()->Column(i)->physical_type());
          break;
      }
      shapes_.push_back(
          TensorShape({static_cast<int64>(parquet_metadata_->num_rows())}));
      dtypes_.push_back(dtype);
      columns_.push_back(
          parquet_metadata_->schema()->Column(i)->path().get()->ToDotString());
      columns_index_[parquet_metadata_->schema()
                         ->Column(i)
                         ->path()
                         .get()
                         ->ToDotString()] = i;
    }
    return Status::OK();
  }

  Status Components(std::vector<string>* components) {
    mutex_lock l(mu_);

    components->clear();
    for (size_t i = 0; i < columns_.size(); i++) {
      components->push_back(columns_[i]);
    }
    return Status::OK();
  }

  Status Spec(const string& component, TensorShape* shape, DataType* dtype) {
    mutex_lock l(mu_);

    if (columns_index_.find(component) == columns_index_.end()) {
      return errors::InvalidArgument("component ", component, " is invalid");
    }
    int64 column_index = columns_index_[component];
    *shape = shapes_[column_index];
    *dtype = dtypes_[column_index];
    return Status::OK();
  }

  Status Read(const string& component,
              const absl::InlinedVector<int64, 4>& start,
              const TensorShape& shape,
              std::function<Status(const TensorShape& shape, Tensor** value)>
                  allocate_func) {
    mutex_lock l(mu_);

    if (columns_index_.find(component) == columns_index_.end()) {
      return errors::InvalidArgument("component ", component, " is invalid");
    }
    const int64 column_index = columns_index_[component];

    Tensor* value;
    TF_RETURN_IF_ERROR(allocate_func(shape, &value));

    const string& column = component;
    int64 element_start = start[0];
    int64 element_stop = start[0] + shape.dim_size(0);

    int64 row_group_offset = 0;
    for (int row_group = 0; row_group < parquet_metadata_->num_row_groups();
         row_group++) {
      std::shared_ptr<parquet::RowGroupReader> row_group_reader =
          parquet_reader_->RowGroup(row_group);
      // Skip if row group is not within [start..stop]
      if ((row_group_offset + row_group_reader->metadata()->num_rows() <
           element_start) ||
          (element_stop <= row_group_offset)) {
        row_group_offset += row_group_reader->metadata()->num_rows();
        continue;
      }
      // Find row_to_read range
      int64 row_to_read_start =
          row_group_offset > element_start ? row_group_offset : element_start;
      int64 row_to_read_final =
          (row_group_offset + row_group_reader->metadata()->num_rows()) <
                  (element_stop)
              ? (row_group_offset + row_group_reader->metadata()->num_rows())
              : (element_stop);
      int64 row_to_read_count = row_to_read_final - row_to_read_start;

      // TODO: parquet is RowGroup based so ideally the RowGroup should be
      // cached with the hope of indexing and slicing happens on each row. For
      // now no caching is done yet.
      std::shared_ptr<parquet::ColumnReader> column_reader =
          row_group_reader->Column(column_index);

      // buffer to fill location is value.data()[row_to_read_start - start]
      // Note: ReadBatch may not be able to read the elements requested
      // (row_to_read_count) in one shot, as such we use while loop of
      // `while (row_left > 0) {...}` to read until complete.

#define PARQUET_PROCESS_TYPE(ptype, type)                                     \
  {                                                                           \
    parquet::TypedColumnReader<ptype>* reader =                               \
        static_cast<parquet::TypedColumnReader<ptype>*>(column_reader.get()); \
    if (row_to_read_start > row_group_offset) {                               \
      reader->Skip(row_to_read_start - row_group_offset);                     \
    }                                                                         \
    ptype::c_type* value_p = (ptype::c_type*)(void*)(&(                       \
        value->flat<type>().data()[row_to_read_start - element_start]));      \
    int64_t row_left = row_to_read_count;                                     \
    while (row_left > 0) {                                                    \
      int64_t values_read;                                                    \
      int64_t levels_read = reader->ReadBatch(                                \
          row_left, nullptr, nullptr, &value_p[row_to_read_count - row_left], \
          &values_read);                                                      \
      if (!(levels_read == values_read && levels_read > 0)) {                 \
        return errors::InvalidArgument("null value in column: ", column);     \
      }                                                                       \
      row_left -= levels_read;                                                \
    }                                                                         \
  }

#define PARQUET_PROCESS_BYTE_ARRAY(ptype)                                     \
  {                                                                           \
    parquet::TypedColumnReader<ptype>* reader =                               \
        static_cast<parquet::TypedColumnReader<ptype>*>(column_reader.get()); \
    if (row_to_read_start > row_group_offset) {                               \
      reader->Skip(row_to_read_start - row_group_offset);                     \
    }                                                                         \
    std::unique_ptr<ptype::c_type[]> value_p(                                 \
        new ptype::c_type[row_to_read_count]);                                \
    int64_t row_left = row_to_read_count;                                     \
    while (row_left > 0) {                                                    \
      int64_t values_read;                                                    \
      int64_t levels_read = reader->ReadBatch(                                \
          row_left, nullptr, nullptr,                                         \
          &value_p.get()[row_to_read_count - row_left], &values_read);        \
      if (!(levels_read == values_read && levels_read > 0)) {                 \
        return errors::InvalidArgument("null value in column: ", column);     \
      }                                                                       \
      row_left -= levels_read;                                                \
    }                                                                         \
    for (int64_t index = 0; index < row_to_read_count; index++) {             \
      value->flat<tstring>()(row_to_read_start - element_start + index) =     \
          ByteArrayToString(value_p[index]);                                  \
    }                                                                         \
  }

#define PARQUET_PROCESS_FIXED_LEN_BYTE_ARRAY(ptype, len)                      \
  {                                                                           \
    parquet::TypedColumnReader<ptype>* reader =                               \
        static_cast<parquet::TypedColumnReader<ptype>*>(column_reader.get()); \
    if (row_to_read_start > row_group_offset) {                               \
      reader->Skip(row_to_read_start - row_group_offset);                     \
    }                                                                         \
    std::unique_ptr<ptype::c_type[]> value_p(                                 \
        new ptype::c_type[row_to_read_count]);                                \
    int64_t row_left = row_to_read_count;                                     \
    while (row_left > 0) {                                                    \
      int64_t values_read;                                                    \
      int64_t levels_read = reader->ReadBatch(                                \
          row_left, nullptr, nullptr,                                         \
          &value_p.get()[row_to_read_count - row_left], &values_read);        \
      if (!(levels_read == values_read && levels_read > 0)) {                 \
        return errors::InvalidArgument("null value in column: ", column);     \
      }                                                                       \
      row_left -= levels_read;                                                \
    }                                                                         \
    for (int64_t index = 0; index < row_to_read_count; index++) {             \
      value->flat<tstring>()(row_to_read_start - element_start + index) =     \
          string((const char*)value_p[index].ptr, len);                       \
    }                                                                         \
  }

      switch (
          parquet_metadata_->schema()->Column(column_index)->physical_type()) {
        case parquet::Type::BOOLEAN:
          PARQUET_PROCESS_TYPE(parquet::BooleanType, bool);
          break;
        case parquet::Type::INT32:
          PARQUET_PROCESS_TYPE(parquet::Int32Type, int32);
          break;
        case parquet::Type::INT64:
          PARQUET_PROCESS_TYPE(parquet::Int64Type, int64);
          break;
        case parquet::Type::FLOAT:
          PARQUET_PROCESS_TYPE(parquet::FloatType, float);
          break;
        case parquet::Type::DOUBLE:
          PARQUET_PROCESS_TYPE(parquet::DoubleType, double);
          break;
        case parquet::Type::BYTE_ARRAY:
          PARQUET_PROCESS_BYTE_ARRAY(parquet::ByteArrayType);
          break;
        case parquet::Type::FIXED_LEN_BYTE_ARRAY:
          PARQUET_PROCESS_FIXED_LEN_BYTE_ARRAY(
              parquet::FLBAType,
              parquet_metadata_->schema()->Column(column_index)->type_length());
          break;
        default:
          return errors::InvalidArgument("invalid data type: ",
                                         parquet_metadata_->schema()
                                             ->Column(column_index)
                                             ->physical_type());
      }
      row_group_offset += row_group_reader->metadata()->num_rows();
    }
    return Status::OK();
  }
  string DebugString() const override { return "ParquetReadableResource"; }

 protected:
  mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<SizedRandomAccessFile> file_ TF_GUARDED_BY(mu_);
  uint64 file_size_ TF_GUARDED_BY(mu_);
  std::shared_ptr<ArrowRandomAccessFile> parquet_file_ TF_GUARDED_BY(mu_);
  std::unique_ptr<::parquet::ParquetFileReader> parquet_reader_
      TF_GUARDED_BY(mu_);
  std::shared_ptr<::parquet::FileMetaData> parquet_metadata_ TF_GUARDED_BY(mu_);

  std::vector<DataType> dtypes_ TF_GUARDED_BY(mu_);
  std::vector<TensorShape> shapes_ TF_GUARDED_BY(mu_);
  std::vector<string> columns_ TF_GUARDED_BY(mu_);
  std::unordered_map<string, int64> columns_index_ TF_GUARDED_BY(mu_);
};

class ParquetReadableInfoOp
    : public IOResourceOpKernel<ParquetReadableResource> {
 public:
  explicit ParquetReadableInfoOp(OpKernelConstruction* context)
      : IOResourceOpKernel<ParquetReadableResource>(context) {}

  virtual ~ParquetReadableInfoOp() {}

  Status ResourceKernel(OpKernelContext* context,
                        ParquetReadableResource* resource) override {
    std::vector<string> components;
    TF_RETURN_IF_ERROR(resource->Components(&components));

    std::vector<TensorShape> shapes;
    std::vector<DataType> dtypes;

    shapes.resize(components.size());
    dtypes.resize(components.size());

    int64 rank = 0;
    for (size_t i = 0; i < components.size(); i++) {
      TF_RETURN_IF_ERROR(resource->Spec(components[i], &shapes[i], &dtypes[i]));
      if (rank < shapes[i].dims()) {
        rank = shapes[i].dims();
      }
    }

    Tensor* component_tensor = nullptr;
    TF_RETURN_IF_ERROR(context->allocate_output(
        0, TensorShape({static_cast<int64>(components.size())}),
        &component_tensor));
    Tensor* shape_tensor = nullptr;
    TF_RETURN_IF_ERROR(context->allocate_output(
        1, TensorShape({static_cast<int64>(components.size()), rank}),
        &shape_tensor));
    Tensor* dtype_tensor = nullptr;
    TF_RETURN_IF_ERROR(context->allocate_output(
        2, TensorShape({static_cast<int64>(components.size())}),
        &dtype_tensor));

    for (size_t i = 0; i < components.size(); i++) {
      component_tensor->flat<tstring>()(i) = components[i];
      for (int64 j = 0; j < shapes[i].dims(); j++) {
        shape_tensor->matrix<int64>()(i, j) = shapes[i].dim_size(j);
      }
      for (int64 j = shapes[i].dims(); j < rank; j++) {
        shape_tensor->matrix<int64>()(i, j) = -1;
      }
      dtype_tensor->flat<int64>()(i) = dtypes[i];
    }
    return Status::OK();
  }
};

class ParquetReadableReadOp
    : public IOResourceOpKernel<ParquetReadableResource> {
 public:
  explicit ParquetReadableReadOp(OpKernelConstruction* context)
      : IOResourceOpKernel<ParquetReadableResource>(context) {}

  virtual ~ParquetReadableReadOp() {}

  Status ResourceKernel(OpKernelContext* context,
                        ParquetReadableResource* resource) override {
    const Tensor* component_tensor;
    TF_RETURN_IF_ERROR(context->input("component", &component_tensor));
    string component = component_tensor->scalar<tstring>()();

    const Tensor* shape_tensor;
    TF_RETURN_IF_ERROR(context->input("shape", &shape_tensor));
    TensorShape shape(shape_tensor->flat<int64>());

    const Tensor* start_tensor;
    TF_RETURN_IF_ERROR(context->input("start", &start_tensor));
    absl::InlinedVector<int64, 4> start(shape.dims());
    for (int64 i = 0; i < start_tensor->NumElements(); i++) {
      start[i] = start_tensor->flat<int64>()(i);
    }
    for (int64 i = start_tensor->NumElements(); i < shape.dims(); i++) {
      start[i] = 0;
    }

    const Tensor* stop_tensor;
    TF_RETURN_IF_ERROR(context->input("stop", &stop_tensor));
    absl::InlinedVector<int64, 4> stop(stop_tensor->shape().dims());
    for (int64 i = 0; i < stop_tensor->NumElements(); i++) {
      stop[i] = stop_tensor->flat<int64>()(i);
    }
    for (int64 i = stop_tensor->NumElements(); i < shape.dims(); i++) {
      stop[i] = shape.dim_size(i);
    }

    for (int64 i = 0; i < shape.dims(); i++) {
      if (stop[i] < 0 || stop[i] > shape.dim_size(i)) {
        stop[i] = shape.dim_size(i);
      }
      if (start[i] > stop[i]) {
        start[i] = stop[i];
      }
    }
    for (int64 i = 0; i < shape.dims(); i++) {
      shape.set_dim(i, stop[i] - start[i]);
    }
    TF_RETURN_IF_ERROR(resource->Read(
        component, start, shape,
        [&](const TensorShape& shape, Tensor** value) -> Status {
          TF_RETURN_IF_ERROR(context->allocate_output(0, shape, value));
          return Status::OK();
        }));
    return Status::OK();
  }
};

REGISTER_KERNEL_BUILDER(Name("IO>ParquetReadableInfo").Device(DEVICE_CPU),
                        ParquetReadableInfoOp);
REGISTER_KERNEL_BUILDER(Name("IO>ParquetReadableRead").Device(DEVICE_CPU),
                        ParquetReadableReadOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
