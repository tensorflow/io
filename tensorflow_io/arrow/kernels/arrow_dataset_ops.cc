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

#include "arrow/api.h"
#include "arrow/ipc/api.h"
#include "arrow/util/io-util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow_io/core/kernels/io_stream.h"
#include "tensorflow_io/arrow/kernels/arrow_kernels.h"
#include "tensorflow_io/arrow/kernels/arrow_stream_client.h"
#include "tensorflow_io/arrow/kernels/arrow_util.h"

#define CHECK_ARROW(arrow_status)             \
  do {                                        \
    arrow::Status _s = (arrow_status);        \
    if (!_s.ok()) {                           \
      return errors::Internal(_s.ToString()); \
    }                                         \
  } while (false)

namespace tensorflow {
namespace data {

enum ArrowBatchMode {
  BATCH_KEEP_REMAINDER,
  BATCH_DROP_REMAINDER,
  BATCH_AUTO,
};

Status GetBatchModeStr(ArrowBatchMode batch_mode, string* batch_mode_str) {
  switch (batch_mode) {
    case ArrowBatchMode::BATCH_KEEP_REMAINDER:
      *batch_mode_str = "keep_remainder";
      break;
    case ArrowBatchMode::BATCH_DROP_REMAINDER:
      *batch_mode_str = "drop_remainder";
      break;
    case ArrowBatchMode::BATCH_AUTO:
      *batch_mode_str = "auto";
      break;
    default:
      return errors::Internal("Unsupported batch mode: " +
                              std::to_string(batch_mode));
  }
  return Status::OK();
}

Status GetBatchMode(string batch_mode_str, ArrowBatchMode* batch_mode ) {
  if (batch_mode_str == "keep_remainder") {
    *batch_mode = ArrowBatchMode::BATCH_KEEP_REMAINDER;
  } else if (batch_mode_str == "drop_remainder") {
    *batch_mode = ArrowBatchMode::BATCH_DROP_REMAINDER;
  } else if (batch_mode_str == "auto") {
    *batch_mode = ArrowBatchMode::BATCH_AUTO;
  } else {
    return errors::Internal("Unsupported batch mode: " + batch_mode_str);
  }
  return Status::OK();
}


// Check the type of an Arrow column matches expected tensor type
class ArrowColumnTypeChecker : public arrow::TypeVisitor {
 public:
  Status CheckColumnType(std::shared_ptr<arrow::DataType> type,
                         DataType expected_type) {
    expected_type_ = expected_type;

    // First see if complex type handled by visitor
    arrow::Status visit_status = type->Accept(this);
    if (visit_status.ok()) {
      return Status::OK();
    }

    // Check type as a scalar type
    CHECK_ARROW(CheckScalarType(type));
    return Status::OK();
  }

 protected:
  virtual arrow::Status Visit(const arrow::ListType& type) {
    return CheckScalarType(type.value_type());
  }

  // Check scalar types with arrow::adapters::tensorflow
  arrow::Status CheckScalarType(std::shared_ptr<arrow::DataType> scalar_type) {
    DataType converted_type;
    ::tensorflow::Status status = GetTensorFlowType(scalar_type, &converted_type);
    if (!status.ok()) {
      return ::arrow::Status::Invalid(status);
    }
    if (converted_type != expected_type_) {
      return arrow::Status::TypeError(
          "Arrow type mismatch: expected dtype=" +
          std::to_string(expected_type_) + ", but got dtype=" +
          std::to_string(converted_type));
    }
    return arrow::Status::OK();
  }

 private:
  DataType expected_type_;
};

// Convert an element of an Arrow Array to a Tensor
class ArrowConvertTensor : public arrow::ArrayVisitor {
 public:
  ArrowConvertTensor(int64_t row_idx, int64_t batch_size, IteratorContext* ctx)
      : curr_index_(row_idx), curr_batch_size_(batch_size), curr_ctx_(ctx),
        curr_array_length_(-1) {}

  // Convert to a Tensor and append to the output vector
  Status AppendTensor(std::shared_ptr<arrow::Array> array, DataType output_type,
                      std::vector<Tensor>* out_tensors) {
    curr_type_ = output_type;
    out_tensors_ = out_tensors;
    if (array->null_count() != 0) {
      return errors::Internal("Arrow arrays with null values not supported");
    }
    CHECK_ARROW(array->Accept(this));
    return Status::OK();
  }

 protected:
  // Get TensorShape for current elements, accounting for arrays and batch size
  TensorShape GetCurrTensorShape() {

    // Start off as scalar and adjust for batch size and array types
    auto shape = TensorShape({});

    // curr_batch_size_ of 0 indicates 1 record at a time, no batching
    if (curr_batch_size_ > 0) {
      shape.AddDim(curr_batch_size_);
    }

    // curr_array_length_ < 0 indicates a scalar value
    if (curr_array_length_ >= 0) {
      shape.AddDim(curr_array_length_);
    }

    return shape;
  }

  virtual arrow::Status Visit(const arrow::BooleanArray& array) {

    // Create a Tensor of required size
    auto shape = GetCurrTensorShape();
    Tensor tensor(curr_ctx_->allocator({}), curr_type_, shape);

    // Must copy one value at a time because Arrow stores values as bits
    for (int64 i = 0; i < shape.num_elements(); ++i) {
      // NOTE: for Array ListArray, curr_row_idx_ is 0 for element array
      tensor.flat<bool>()(i) = array.Value(i + curr_index_);
    }

    out_tensors_->emplace_back(std::move(tensor));
    return arrow::Status::OK();
  }

  template <typename ArrayType>
  arrow::Status VisitFixedWidth(const ArrayType& array) {
    const auto& fw_type =
        static_cast<const arrow::FixedWidthType&>(*array.type());
    const int64_t type_width = fw_type.bit_width() / 8;

    // Create a Tensor of required size
    auto shape = GetCurrTensorShape();
    Tensor tensor(curr_ctx_->allocator({}), curr_type_, shape);

    // Primitive Arrow arrays have validity and value buffers, currently
    // only arrays with null count == 0 are supported, so only need values here
    static const int VALUE_BUFFER = 1;
    auto values = array.data()->buffers[VALUE_BUFFER];
    if (values == NULLPTR) {
      return arrow::Status::Invalid(
          "Received an Arrow array with a NULL value buffer");
    }

    const void* src = (values->data() + array.data()->offset * type_width) +
                      curr_index_ * type_width;
    void* dst = const_cast<char*>(tensor.tensor_data().data());
    std::memcpy(dst, src, shape.num_elements() * type_width);

    out_tensors_->emplace_back(std::move(tensor));
    return arrow::Status::OK();
  }

#define VISIT_FIXED_WIDTH(TYPE)                             \
  virtual arrow::Status Visit(const TYPE& array) override { \
    return VisitFixedWidth(array);                          \
  }

  VISIT_FIXED_WIDTH(arrow::Int8Array)
  VISIT_FIXED_WIDTH(arrow::Int16Array)
  VISIT_FIXED_WIDTH(arrow::Int32Array)
  VISIT_FIXED_WIDTH(arrow::Int64Array)
  VISIT_FIXED_WIDTH(arrow::UInt8Array)
  VISIT_FIXED_WIDTH(arrow::UInt16Array)
  VISIT_FIXED_WIDTH(arrow::UInt32Array)
  VISIT_FIXED_WIDTH(arrow::UInt64Array)
  VISIT_FIXED_WIDTH(arrow::HalfFloatArray)
  VISIT_FIXED_WIDTH(arrow::FloatArray)
  VISIT_FIXED_WIDTH(arrow::DoubleArray)
#undef VISIT_FIXED_WITH

  virtual arrow::Status Visit(const arrow::ListArray& array) override {
    int32 values_offset = array.value_offset(curr_index_);
    curr_array_length_ = array.value_length(curr_index_);
    int32 num_arrays = 1;

    // If batching tensors, arrays must be same length
    if (curr_batch_size_ > 0) {
      num_arrays = curr_batch_size_;
      for (int64_t i = curr_index_; i < curr_index_ + num_arrays; ++i) {
        if (array.value_length(i) != curr_array_length_) {
          return arrow::Status::Invalid(
              "Batching variable-length arrays is unsupported");
        }
      }
    }

    // Save current index and swap after array is copied
    int32 tmp_index = curr_index_;
    curr_index_ = 0;

    // Prepare the array data buffer and visit the array slice
    std::shared_ptr<arrow::Array> values = array.values();
    std::shared_ptr<arrow::Array> element_values =
        values->Slice(values_offset, curr_array_length_ * num_arrays);
    auto result = element_values->Accept(this);

    // Reset state variables for next time
    curr_index_ = tmp_index;
    curr_array_length_ = -1;
    return result;
  }

 private:
  int64_t curr_index_;
  int64_t curr_batch_size_;
  IteratorContext* curr_ctx_;
  int32_t curr_array_length_;
  DataType curr_type_;
  std::vector<Tensor>* out_tensors_;
};

// Base class for defining a Dataset over Arrow record batches with an
// iterator that iterates over rows of the batch to get Tensors
class ArrowDatasetBase : public DatasetBase {
 public:
  ArrowDatasetBase(OpKernelContext* ctx,
                   const std::vector<int32>& columns,
                   const int64 batch_size,
                   const ArrowBatchMode batch_mode,
                   const DataTypeVector& output_types,
                   const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        columns_(columns),
        batch_size_(batch_size),
        batch_mode_(batch_mode),
        output_types_(output_types),
        output_shapes_(output_shapes) {}

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

 protected:
  // Abstract base class for iterating over rows of Arrow record
  // batches. Implementations will define how record batches are
  // initialized and consumed.
  template <typename DatasetType>
  class ArrowBaseIterator : public DatasetIterator<DatasetType> {
   public:
    ArrowBaseIterator(
        const typename DatasetIterator<DatasetType>::Params& params)
        : DatasetIterator<DatasetType>(params) {}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);

      // If in initial state, setup and read first batch
      if (current_batch_ == nullptr && current_row_idx_ == 0) {
        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      }

      std::vector<Tensor>* result_tensors = out_tensors;
      auto partial_batches =
          std::vector<std::shared_ptr<std::vector<Tensor>>>();
      int64 partial_batch_size = 0;
      bool have_result = false;

      // Loop until have_result or end_of_sequence
      do {

      // Try to go to next batch if consumed all rows in current batch
      if (current_batch_ != nullptr &&
          current_row_idx_ >= current_batch_->num_rows()) {
        TF_RETURN_IF_ERROR(NextStreamLocked(ctx->env()));
      }

      // Check if reached end of stream
      if (current_batch_ == nullptr) {

        // Finalize the iterator state
        ResetStreamsLocked();

        // Return partial batch if drop_remainder flag not set
        if (partial_batch_size > 0 && this->dataset()->batch_mode_ !=
            ArrowBatchMode::BATCH_DROP_REMAINDER) {
          // Copy partial batched tensors to output tensors
          TF_RETURN_IF_ERROR(AppendPartialTensors(ctx, partial_batch_size,
              partial_batches, out_tensors));
          have_result = true;
        // No more results, so end the sequence
        }  else {
          *end_of_sequence = true;
        }
      } else {

        // Calc the batch size, will be 0 if not batching
        int64 batch_size =
            this->dataset()->batch_mode_ == ArrowBatchMode::BATCH_AUTO ?
                // Auto batch size is number of rows in current record batch
                current_batch_->num_rows() :
                // Use set batch size minus any partials already read
                this->dataset()->batch_size_ - partial_batch_size;

        // Prepare a partial batch to save, either current record batch is too
        // small or continuing to fill previous partial batch
        if (batch_size != 0 && (partial_batch_size > 0 ||
            current_row_idx_ + batch_size > current_batch_->num_rows())) {
          int64 rows_remaining = current_batch_->num_rows() - current_row_idx_;
          batch_size = std::min(batch_size, rows_remaining);
          partial_batches.push_back(std::make_shared<std::vector<Tensor>>());
          result_tensors = partial_batches.back().get();
          partial_batch_size += batch_size;
        }

        // Assign Tensors for each column in the current row
        ArrowConvertTensor arrow_converter(current_row_idx_, batch_size, ctx);
        for (size_t i = 0; i < this->dataset()->columns_.size(); ++i) {
          int32 col = this->dataset()->columns_[i];
          DataType dt = this->dataset()->output_types_[i];
          std::shared_ptr<arrow::Array> arr = current_batch_->column(col);
          TF_RETURN_IF_ERROR(
              arrow_converter.AppendTensor(arr, dt, result_tensors));
        }

        // If not batching or have a full batch, then have a result to return
        if (partial_batch_size == 0 ||
            partial_batch_size == this->dataset()->batch_size_) {
          have_result = true;

          // If have partial batches, copy partial tensors to output tensors
          if (!partial_batches.empty()) {
            TF_RETURN_IF_ERROR(AppendPartialTensors(ctx, partial_batch_size,
                  partial_batches, out_tensors));
          }
        }

        // Increment to next row or batch
        current_row_idx_ += batch_size == 0 ? 1 : batch_size;
        *end_of_sequence = false;
      }

      } while (!(have_result || *end_of_sequence));

      return Status::OK();
    }

   private:
    Status AppendPartialTensors(
        IteratorContext* ctx,
        int64 batch_size,
        const std::vector<std::shared_ptr<std::vector<Tensor>>>& partials,
        std::vector<Tensor>* out_tensors) {
      int64 batch_index = 0;

      // If only one partial batch, can just move to output
      if (partials.size() == 1) {
        *out_tensors = std::move(*partials.at(0).get());
        return Status::OK();
      }

      // Copy all partial tensors to a single output tensor
      for (auto it_partial = partials.begin(); it_partial != partials.end();
          it_partial++) {
        int64 partial_batch_size = 0;
        for (size_t i = 0; i < (*it_partial)->size(); ++i) {
          const Tensor &element = (*it_partial)->at(i);
          partial_batch_size = element.dim_size(0);

          // Allocate tensor sized to batch on first iteration
          if (it_partial == partials.begin()) {
            TensorShape shape = element.shape();
            shape.set_dim(0, batch_size);
            Tensor output(ctx->allocator({}), element.dtype(), shape);
            out_tensors->emplace_back(std::move(output));
          }

          // Copy partial batch to the output batch
          TF_RETURN_IF_ERROR(
              CopyElementsToParent(element, &out_tensors->at(i), batch_index));
        }
        batch_index += partial_batch_size;
      }
      return Status::OK();
    }

    template <typename T>
    Status HandleElementsToParent(
        const Tensor& element, Tensor* parent, int64 index) {
      // TODO: look into removing this loop, move tensor instead of copy
      for (int64 i = 0; i < element.dim_size(0); ++i) {
        parent->flat_outer_dims<T>().chip(index + i, 0) =
            element.flat_outer_dims<T>().chip(i, 0);
      }
      return Status::OK();
    }

    Status CopyElementsToParent(
        const Tensor& element,
        Tensor *parent,
        int64 index) {
#define HANDLE_TYPE(T)                                                       \
      case DataTypeToEnum<T>::value: {                                       \
        return HandleElementsToParent<T>(std::move(element), parent, index); \
      }

      switch (element.dtype()) {
        TF_CALL_ALL_TYPES(HANDLE_TYPE);
        TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
        TF_CALL_uint32(HANDLE_TYPE);
        TF_CALL_uint64(HANDLE_TYPE);
#undef HANDLE_TYPE
        default:
          return errors::Unimplemented(
              "CopyElementsToParent Unhandled data type: ", element.dtype());
      }
    }

   protected:
    Status SaveInternal(IteratorStateWriter* writer) override {
      return errors::Unimplemented("SaveInternal is currently not supported");
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return errors::Unimplemented(
          "RestoreInternal is currently not supported");
    }

    // Setup Arrow record batch consumer and initialze current_batch_
    virtual Status SetupStreamsLocked(Env* env)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

    // Get the next Arrow record batch, if available. If not then
    // current_batch_ will be set to nullptr to indicate no further batches.
    virtual Status NextStreamLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      current_batch_ = nullptr;
      current_row_idx_ = 0;
      return Status::OK();
    }

    // Reset the Arrow record batch consumer when done with batches.
    virtual void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      // This is the final state of the iterator after end_of_sequence=true
      current_batch_ = nullptr;
      current_row_idx_ = 1;
    }

    // Check columns of batch in stream are expected data type
    Status CheckBatchColumnTypes(std::shared_ptr<arrow::RecordBatch> batch) {
      ArrowColumnTypeChecker type_checker;
      for (size_t i = 0; i < this->dataset()->columns_.size(); ++i) {
        int32 col = this->dataset()->columns_[i];
        DataType dt = this->dataset()->output_types_[i];
        std::shared_ptr<arrow::Array> arr = batch->column(col);
        TF_RETURN_IF_ERROR(type_checker.CheckColumnType(arr->type(), dt));
      }
      return Status::OK();
    }

    mutex mu_;
    std::shared_ptr<arrow::RecordBatch> current_batch_ GUARDED_BY(mu_) =
        nullptr;
    int64_t current_row_idx_ GUARDED_BY(mu_) = 0;
  };

  const std::vector<int32> columns_;
  const int64 batch_size_;
  const ArrowBatchMode batch_mode_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

// Abstract base class to define an Arrow OpKernel with output_types and
// output_shapes attributes, and list of column indices. Implementations
// will define how to create the Arrow Dataset.
class ArrowOpKernelBase : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  ArrowOpKernelBase(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    for (const DataType& dt : output_types_) {
      std::shared_ptr<arrow::DataType> arrow_type;
      OP_REQUIRES_OK(ctx, GetArrowType(dt, &arrow_type));
    }
    for (const PartialTensorShape& pts : output_shapes_) {
      OP_REQUIRES(ctx, -1 <= pts.dims() && pts.dims() <= 2,
                  errors::InvalidArgument("Output shape must be a scalar, "
                                          "vector, matrix or unknown"));
    }
  }

 private:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* columns_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("columns", &columns_tensor));
    OP_REQUIRES(
        ctx, columns_tensor->dims() <= 1,
        errors::InvalidArgument("`columns` must be a scalar or a vector."));

    std::vector<int32> columns;
    columns.reserve(columns_tensor->NumElements());
    for (int32 i = 0; i < static_cast<int32>(columns_tensor->NumElements());
         ++i) {
      columns.push_back(columns_tensor->flat<int32>()(i));
    }

    int64 batch_size;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "batch_size", &batch_size));

    string batch_mode_str;
    OP_REQUIRES_OK(ctx,
        ParseScalarArgument(ctx, "batch_mode", &batch_mode_str));
    ArrowBatchMode batch_mode;
    OP_REQUIRES_OK(ctx, GetBatchMode(batch_mode_str, &batch_mode));

    ArrowDatasetBase* arrow_output;
    MakeArrowDataset(
        ctx,
        columns,
        batch_size,
        batch_mode,
        output_types_,
        output_shapes_,
        &arrow_output);
    *output = arrow_output;
  }

 protected:
  // Define to construct an implementation of ArrowDatasetBase
  virtual void MakeArrowDataset(
      OpKernelContext* ctx,
      const std::vector<int32>& columns,
      const int64 batch_size,
      const ArrowBatchMode batch_mode,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      ArrowDatasetBase** output) = 0;

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

// Op to create an ArrowZeroCopyDataset that consumes Arrow record batches
// from a memory buffer address owned in Python.
class ArrowZeroCopyDatasetOp : public ArrowOpKernelBase {
 public:
  explicit ArrowZeroCopyDatasetOp(OpKernelConstruction* ctx)
      : ArrowOpKernelBase(ctx) {}

  virtual void MakeArrowDataset(
      OpKernelContext* ctx,
      const std::vector<int32>& columns,
      const int64 batch_size,
      const ArrowBatchMode batch_mode,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      ArrowDatasetBase** output) override {
    uintptr_t buffer_address;
    OP_REQUIRES_OK(
        ctx,
        ParseScalarArgument<uintptr_t>(ctx, "buffer_address", &buffer_address));
    const uint8_t* buffer = reinterpret_cast<const uint8_t*>(buffer_address);

    int64_t buffer_size;
    OP_REQUIRES_OK(
        ctx,
        ParseScalarArgument<int64_t>(ctx, "buffer_size", &buffer_size));
    *output = new Dataset(
        ctx,
        buffer,
        buffer_size,
        columns,
        batch_size,
        batch_mode,
        output_types_,
        output_shapes_);
  }

 private:
  class Dataset : public ArrowDatasetBase {
   public:
    Dataset(OpKernelContext* ctx,
            const uint8_t* buffer_ptr,
            const int64 buffer_size,
            const std::vector<int32>& columns,
            const int64 batch_size,
            const ArrowBatchMode batch_mode,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : ArrowDatasetBase(ctx, columns, batch_size, batch_mode,
              output_types, output_shapes),
          buffer_ptr_(buffer_ptr), buffer_size_(buffer_size) {}

    string DebugString() const override {
      return "ArrowZeroCopyDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* buffer = nullptr;
      uintptr_t buffer_temp = reinterpret_cast<uintptr_t>(buffer_ptr_);
      uint64 buffer_address = buffer_temp;
      TF_RETURN_IF_ERROR(b->AddScalar(buffer_address, &buffer));
      Node* size = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddScalar(static_cast<int64>(buffer_size_), &size));
      Node* columns = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(columns_, &columns));
      Node* batch_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));
      Node* batch_mode = nullptr;
      string batch_mode_str;
      TF_RETURN_IF_ERROR(GetBatchModeStr(batch_mode_, &batch_mode_str));
      TF_RETURN_IF_ERROR(b->AddScalar(batch_mode_str, &batch_mode));
      TF_RETURN_IF_ERROR(
          b->AddDataset(
            this,
            {buffer, size, columns, batch_size, batch_mode},
            output));
      return Status::OK();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Arrow")}));
    }

   private:
    class Iterator : public ArrowBaseIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : ArrowBaseIterator<Dataset>(params) {}

     private:
      Status SetupStreamsLocked(Env* env)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        buffer_ = std::make_shared<arrow::Buffer>(
            dataset()->buffer_ptr_,
            dataset()->buffer_size_);
        buffer_reader_ = std::make_shared<arrow::io::BufferReader>(buffer_);
        CHECK_ARROW(
            arrow::ipc::RecordBatchFileReader::Open(
              buffer_reader_.get(),
              buffer_->size(),
              &reader_));
        num_batches_ = reader_->num_record_batches();
        if (num_batches_ > 0) {
          CHECK_ARROW(
              reader_->ReadRecordBatch(current_batch_idx_, &current_batch_));
          TF_RETURN_IF_ERROR(CheckBatchColumnTypes(current_batch_));
        }
        return Status::OK();
      }

      Status NextStreamLocked(Env* env)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::NextStreamLocked(env);
        if (++current_batch_idx_ < num_batches_) {
          CHECK_ARROW(
              reader_->ReadRecordBatch(current_batch_idx_, &current_batch_));
        }
        return Status::OK();
      }

      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::ResetStreamsLocked();
        reader_.reset();
        current_batch_idx_ = 0;
        num_batches_ = 0;
      }

      std::shared_ptr<arrow::Buffer> buffer_ GUARDED_BY(mu_);
      std::shared_ptr<arrow::io::BufferReader> buffer_reader_ GUARDED_BY(mu_);
      std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader_
          GUARDED_BY(mu_);
      int current_batch_idx_ GUARDED_BY(mu_) = 0;
      int num_batches_ GUARDED_BY(mu_) = 0;
    };

    const uint8_t* buffer_ptr_;
    const int64 buffer_size_;
  };
};

// Op to create an ArrowSerializedDataset that consumes Arrow record batches
// serialized in a Tensor buffer.
class ArrowSerializedDatasetOp : public ArrowOpKernelBase {
 public:
  explicit ArrowSerializedDatasetOp(OpKernelConstruction* ctx)
      : ArrowOpKernelBase(ctx) {}

  virtual void MakeArrowDataset(
      OpKernelContext* ctx,
      const std::vector<int32>& columns,
      const int64 batch_size,
      const ArrowBatchMode batch_mode,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      ArrowDatasetBase** output) override {
    const Tensor* batches_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("serialized_batches", &batches_tensor));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(batches_tensor->shape()),
        errors::InvalidArgument("serialized_batches must be a scalar"));
    *output = new Dataset(
        ctx,
        *batches_tensor,
        columns,
        batch_size,
        batch_mode,
        output_types_,
        output_shapes_);
  }

 private:
  class Dataset : public ArrowDatasetBase {
   public:
    // Construct a Dataset that consumed Arrow batches from serialized bytes
    // in a string. Record batches should be serialized in Arrow File format.
    Dataset(OpKernelContext* ctx,
            const Tensor batches_tensor,
            const std::vector<int32>& columns,
            const int64 batch_size,
            const ArrowBatchMode batch_mode,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : ArrowDatasetBase(ctx, columns, batch_size, batch_mode,
              output_types, output_shapes),
          batches_(std::move(batches_tensor)) {
    }

    string DebugString() const override {
      return "ArrowSerializedDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* batches = nullptr;
      // optimization_only has been removed in
      // https://github.com/tensorflow/tensorflow/commit/6d8f05acd72df61e5f4e5b4c72837b7caed3e942#diff-5eac6c133a3a701a696767960e796bd3
      //if (ctx->optimization_only()) {
      //  TF_RETURN_IF_ERROR(b->AddPlaceholder(batches_, &batches));
      //  DCHECK_NE(ctx->input_list(), nullptr);
      //  ctx->input_list()->emplace_back(batches->name(), batches_);
      //} else {
      //  TF_RETURN_IF_ERROR(b->AddTensor(batches_, &batches));
      //}
      TF_RETURN_IF_ERROR(b->AddTensor(batches_, &batches));
      Node* columns = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(columns_, &columns));
      Node* batch_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));
      Node* batch_mode = nullptr;
      string batch_mode_str;
      TF_RETURN_IF_ERROR(GetBatchModeStr(batch_mode_, &batch_mode_str));
      TF_RETURN_IF_ERROR(b->AddScalar(batch_mode_str, &batch_mode));
      TF_RETURN_IF_ERROR(
          b->AddDataset(
            this,
            {batches, columns, batch_size, batch_mode},
            output));
      return Status::OK();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Arrow")}));
    }

   private:
    class Iterator : public ArrowBaseIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : ArrowBaseIterator<Dataset>(params) {}

     private:
      Status SetupStreamsLocked(Env* env)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        const string& batches = dataset()->batches_.scalar<string>()();
        auto buffer = std::make_shared<arrow::Buffer>(batches);
        auto buffer_reader = std::make_shared<arrow::io::BufferReader>(buffer);
        CHECK_ARROW(
            arrow::ipc::RecordBatchFileReader::Open(buffer_reader, &reader_));
        num_batches_ = reader_->num_record_batches();
        if (num_batches_ > 0) {
          CHECK_ARROW(
              reader_->ReadRecordBatch(current_batch_idx_, &current_batch_));
          TF_RETURN_IF_ERROR(CheckBatchColumnTypes(current_batch_));
        }
        return Status::OK();
      }

      Status NextStreamLocked(Env* env)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::NextStreamLocked(env);
        if (++current_batch_idx_ < num_batches_) {
          CHECK_ARROW(
              reader_->ReadRecordBatch(current_batch_idx_, &current_batch_));
        }
        return Status::OK();
      }

      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::ResetStreamsLocked();
        reader_.reset();
        current_batch_idx_ = 0;
        num_batches_ = 0;
      }

      std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader_
          GUARDED_BY(mu_);
      int current_batch_idx_ GUARDED_BY(mu_) = 0;
      int num_batches_ GUARDED_BY(mu_) = 0;
    };

    const Tensor batches_;
  };
};

// Op to create an Arrow Dataset that consumes record batches from a list of
// files in Arrow Feather format. Feather is a light-weight columnar format
// ideal for simple writing of Pandas DataFrames.
class ArrowFeatherDatasetOp : public ArrowOpKernelBase {
 public:
  explicit ArrowFeatherDatasetOp(OpKernelConstruction* ctx)
      : ArrowOpKernelBase(ctx) {}

  virtual void MakeArrowDataset(
      OpKernelContext* ctx,
      const std::vector<int32>& columns,
      const int64 batch_size,
      const ArrowBatchMode batch_mode,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      ArrowDatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or vector."));
    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }

    *output = new Dataset(
        ctx,
        filenames,
        columns,
        batch_size,
        batch_mode,
        output_types_,
        output_shapes_);
  }

 private:
  class Dataset : public ArrowDatasetBase {
   public:
    Dataset(OpKernelContext* ctx,
            const std::vector<string>& filenames,
            const std::vector<int32>& columns,
            const int64 batch_size,
            const ArrowBatchMode batch_mode,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : ArrowDatasetBase(ctx, columns, batch_size, batch_mode,
              output_types, output_shapes),
          filenames_(filenames) {}

    string DebugString() const override {
      return "ArrowFeatherDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      Node* columns = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(columns_, &columns));
      Node* batch_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));
      Node* batch_mode = nullptr;
      string batch_mode_str;
      TF_RETURN_IF_ERROR(GetBatchModeStr(batch_mode_, &batch_mode_str));
      TF_RETURN_IF_ERROR(b->AddScalar(batch_mode_str, &batch_mode));
      TF_RETURN_IF_ERROR(
          b->AddDataset(
            this,
            {filenames, columns, batch_size, batch_mode},
            output));
      return Status::OK();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::ArrowFeather")}));
    }

   private:
    class Iterator : public ArrowBaseIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : ArrowBaseIterator<Dataset>(params) {}

     private:
      Status SetupStreamsLocked(Env* env)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        const string& filename = dataset()->filenames_[current_file_idx_];

        // Init a TF file from the filename and determine size
        // TODO: set optional memory to nullptr until input arg is added
        std::shared_ptr<SizedRandomAccessFile> tf_file(
            new SizedRandomAccessFile(env, filename, nullptr, 0));
        uint64 size;
        TF_RETURN_IF_ERROR(tf_file->GetFileSize(&size));

        // Wrap the TF file in Arrow interface to be used in Feather reader
        std::shared_ptr<ArrowRandomAccessFile> in_file(
            new ArrowRandomAccessFile(tf_file.get(), size));

        // Create the Feather reader
        std::unique_ptr<arrow::ipc::feather::TableReader> reader;
        CHECK_ARROW(arrow::ipc::feather::TableReader::Open(in_file, &reader));

        // Read file columns and build a table
        int64_t num_columns = reader->num_columns();
        std::vector<std::shared_ptr<arrow::Field>> fields(num_columns);
        std::vector<std::shared_ptr<arrow::Column>> columns(num_columns);
        for (int64_t i = 0; i < num_columns; ++i) {
          CHECK_ARROW(reader->GetColumn(i, &columns[i]));
          fields[i] = columns[i]->field();
        }
        auto schema = std::make_shared<arrow::Schema>(fields);
        auto table = arrow::Table::Make(schema, columns);

        // Convert the table to a sequence of batches
        arrow::TableBatchReader tr(*table.get());
        std::shared_ptr<arrow::RecordBatch> batch;
        CHECK_ARROW(tr.ReadNext(&batch));
        TF_RETURN_IF_ERROR(CheckBatchColumnTypes(batch));
        current_batch_ = batch;
        while (batch != nullptr) {
          record_batches_.push_back(batch);
          CHECK_ARROW(tr.ReadNext(&batch));
        }
        return Status::OK();
      }

      Status NextStreamLocked(Env* env)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::NextStreamLocked(env);
        if (++current_batch_idx_ < record_batches_.size()) {
          current_batch_ = record_batches_[current_batch_idx_];
        } else if (++current_file_idx_ < dataset()->filenames_.size()) {
          current_batch_idx_ = 0;
          record_batches_.clear();
          return SetupStreamsLocked(env);
        }
        return Status::OK();
      }

      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::ResetStreamsLocked();
        current_file_idx_ = 0;
        current_batch_idx_ = 0;
        record_batches_.clear();
      }

      size_t current_file_idx_ GUARDED_BY(mu_) = 0;
      size_t current_batch_idx_ GUARDED_BY(mu_) = 0;
      std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches_
          GUARDED_BY(mu_);
    };

    const std::vector<string> filenames_;
  };
};

// Op to create an Arrow Dataset that consumes record batches from an input
// stream. Currently supported endpoints are a POSIX IPv4 socket with endpoint
// "<IP>:<PORT>" or "tcp://<IP>:<PORT>", a Unix Domain Socket with endpoint
// "unix://<pathname>", and STDIN with endpoint "fd://0" or "fd://-".
class ArrowStreamDatasetOp : public ArrowOpKernelBase {
 public:
  explicit ArrowStreamDatasetOp(OpKernelConstruction* ctx)
      : ArrowOpKernelBase(ctx) {}

  virtual void MakeArrowDataset(
      OpKernelContext* ctx,
      const std::vector<int32>& columns,
      const int64 batch_size,
      const ArrowBatchMode batch_mode,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      ArrowDatasetBase** output) override {
    const Tensor* endpoints_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("endpoints", &endpoints_tensor));
    OP_REQUIRES(
        ctx, endpoints_tensor->dims() <= 1,
        errors::InvalidArgument("`endpoints` must be a scalar or vector."));
    std::vector<string> endpoints;
    endpoints.reserve(endpoints_tensor->NumElements());
    for (int i = 0; i < endpoints_tensor->NumElements(); ++i) {
      endpoints.push_back(endpoints_tensor->flat<string>()(i));
    }

    *output = new Dataset(
        ctx,
        endpoints,
        columns,
        batch_size,
        batch_mode,
        output_types_,
        output_shapes_);
  }

 private:
  class Dataset : public ArrowDatasetBase {
   public:
    Dataset(OpKernelContext* ctx,
            const std::vector<string>& endpoints,
            const std::vector<int32>& columns,
            const int64 batch_size,
            const ArrowBatchMode batch_mode,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : ArrowDatasetBase(ctx, columns, batch_size, batch_mode,
              output_types, output_shapes),
          endpoints_(endpoints) {}

    string DebugString() const override {
      return "ArrowStreamDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* endpoints = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(endpoints_, &endpoints));
      Node* columns = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(columns_, &columns));
      Node* batch_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));
      Node* batch_mode = nullptr;
      string batch_mode_str;
      TF_RETURN_IF_ERROR(GetBatchModeStr(batch_mode_, &batch_mode_str));
      TF_RETURN_IF_ERROR(b->AddScalar(batch_mode_str, &batch_mode));
      TF_RETURN_IF_ERROR(
          b->AddDataset(
            this,
            {endpoints, columns, batch_size, batch_mode},
            output));
      return Status::OK();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::ArrowStream")}));
    }

   private:
    class Iterator : public ArrowBaseIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : ArrowBaseIterator<Dataset>(params) {}

     private:
      Status SetupStreamsLocked(Env* env)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        const string& endpoint = dataset()->endpoints_[current_endpoint_idx_];
        string endpoint_type;
        string endpoint_value;
        TF_RETURN_IF_ERROR(
            ParseEndpoint(endpoint, &endpoint_type, &endpoint_value));

        // Check if endpoint is STDIN
        if (endpoint_type == "fd" && (endpoint_value == "0" ||
                                      endpoint_value == "-")) {
          in_stream_ = std::make_shared<arrow::io::StdinStream>();
        } else {
          // Endpoint is a socket, make a client connection
          auto socket_stream = std::make_shared<ArrowStreamClient>(endpoint);
          CHECK_ARROW(socket_stream->Connect());
          in_stream_ = socket_stream;
        }

        CHECK_ARROW(arrow::ipc::RecordBatchStreamReader::Open(in_stream_.get(),
                                                              &reader_));
        CHECK_ARROW(reader_->ReadNext(&current_batch_));
        TF_RETURN_IF_ERROR(CheckBatchColumnTypes(current_batch_));
        return Status::OK();
      }

      Status NextStreamLocked(Env* env)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::NextStreamLocked(env);
        CHECK_ARROW(reader_->ReadNext(&current_batch_));
        if (current_batch_ == nullptr &&
            ++current_endpoint_idx_ < dataset()->endpoints_.size()) {
          reader_.reset();
          SetupStreamsLocked(env);
        }
        return Status::OK();
      }

      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::ResetStreamsLocked();
        current_endpoint_idx_ = 0;
        reader_.reset();
        in_stream_.reset();
      }

      size_t current_endpoint_idx_ GUARDED_BY(mu_) = 0;
      std::shared_ptr<arrow::io::InputStream> in_stream_ GUARDED_BY(mu_);
      std::shared_ptr<arrow::ipc::RecordBatchReader> reader_ GUARDED_BY(mu_);
    };

    const std::vector<string> endpoints_;
  };
};

REGISTER_KERNEL_BUILDER(Name("IO>ArrowZeroCopyDataset").Device(DEVICE_CPU),
                        ArrowZeroCopyDatasetOp);

REGISTER_KERNEL_BUILDER(Name("IO>ArrowSerializedDataset").Device(DEVICE_CPU),
                        ArrowSerializedDatasetOp);

REGISTER_KERNEL_BUILDER(Name("IO>ArrowFeatherDataset").Device(DEVICE_CPU),
                        ArrowFeatherDatasetOp);

REGISTER_KERNEL_BUILDER(Name("IO>ArrowStreamDataset").Device(DEVICE_CPU),
                        ArrowStreamDatasetOp);

}  // namespace data
}  // namespace tensorflow
