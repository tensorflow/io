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

#ifndef TENSORFLOW_IO_BIGQUERY_KERNELS_BIGQUERY_LIB_H_
#define TENSORFLOW_IO_BIGQUERY_KERNELS_BIGQUERY_LIB_H_

#include <grpcpp/grpcpp.h>

#include "api/Compiler.hh"
#include "api/DataFile.hh"
#include "api/Decoder.hh"
#include "api/Encoder.hh"
#include "api/Generic.hh"
#include "api/Specific.hh"
#include "api/ValidSchema.hh"
#include "arrow/api.h"
#include "arrow/buffer.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/api.h"
#include "google/cloud/bigquery/storage/v1beta1/storage.grpc.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_io/arrow/kernels/arrow_util.h"

namespace tensorflow {

namespace apiv1beta1 = ::google::cloud::bigquery::storage::v1beta1;

Status GrpcStatusToTfStatus(const ::grpc::Status &status);
string GrpcStatusToString(const ::grpc::Status &status);
Status GetDataFormat(string data_format_str,
                     apiv1beta1::DataFormat *data_format);

class BigQueryClientResource : public ResourceBase {
 public:
  explicit BigQueryClientResource(
      std::shared_ptr<apiv1beta1::BigQueryStorage::Stub> stub)
      : stub_(std::move(stub)) {}

  std::shared_ptr<apiv1beta1::BigQueryStorage::Stub> get_stub() {
    return stub_;
  }

  string DebugString() const override { return "BigQueryClientResource"; }

 private:
  std::shared_ptr<apiv1beta1::BigQueryStorage::Stub> stub_;
};

namespace data {

// BigQueryReaderDatasetIteratorBase is an abstract class for iterators from
// datasets that are "readers" (source datasets, not transformation datasets)
// that read from BigQuery.
template <typename Dataset>
class BigQueryReaderDatasetIteratorBase : public DatasetIterator<Dataset> {
 public:
  Status GetNextInternal(IteratorContext *ctx, std::vector<Tensor> *out_tensors,
                         bool *end_of_sequence) override {
    mutex_lock l(mu_);
    VLOG(3)
        << "calling BigQueryReaderDatasetIteratorBase.GetNextInternal() index: "
        << current_row_index_ << " stream: " << this->dataset()->stream();
    *end_of_sequence = false;

    TF_RETURN_IF_ERROR(EnsureReaderInitialized());
    TF_RETURN_IF_ERROR(EnsureHasRow(end_of_sequence));
    if (*end_of_sequence) {
      VLOG(3) << "end of sequence";
      return Status::OK();
    }

    auto status =
        ReadRecord(ctx, out_tensors, this->dataset()->selected_fields(),
                   this->dataset()->output_types());
    current_row_index_++;
    return status;
  }

 protected:
  explicit BigQueryReaderDatasetIteratorBase(
      const typename DatasetIterator<Dataset>::Params &params)
      : DatasetIterator<Dataset>(params) {
    VLOG(3) << "created BigQueryReaderDatasetIteratorBase for stream: "
            << this->dataset()->stream();
  }

  virtual Status EnsureReaderInitialized() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (reader_) {
      return Status::OK();
    }

    apiv1beta1::ReadRowsRequest readRowsRequest;
    readRowsRequest.mutable_read_position()->mutable_stream()->set_name(
        this->dataset()->stream());
    readRowsRequest.mutable_read_position()->set_offset(
        this->dataset()->offset());

    read_rows_context_ = absl::make_unique<::grpc::ClientContext>();
    read_rows_context_->AddMetadata(
        "x-goog-request-params",
        absl::StrCat("read_position.stream.name=",
                     readRowsRequest.read_position().stream().name()));

    VLOG(3) << "getting reader, stream: "
            << readRowsRequest.read_position().stream().DebugString();
    reader_ = this->dataset()->client_resource()->get_stub()->ReadRows(
        read_rows_context_.get(), readRowsRequest);

    return Status::OK();
  }

  virtual Status EnsureHasRow(bool *end_of_sequence) = 0;
  virtual Status ReadRecord(IteratorContext *ctx,
                            std::vector<Tensor> *out_tensors,
                            const std::vector<string> &columns,
                            const std::vector<DataType> &output_types) = 0;
  int current_row_index_ = 0;
  mutex mu_;
  std::unique_ptr<::grpc::ClientContext> read_rows_context_ TF_GUARDED_BY(mu_);
  std::unique_ptr<::grpc::ClientReader<apiv1beta1::ReadRowsResponse>> reader_
      TF_GUARDED_BY(mu_);
  std::unique_ptr<apiv1beta1::ReadRowsResponse> response_ TF_GUARDED_BY(mu_);
};

// BigQuery reader for Arrow serialized data.
template <typename Dataset>
class BigQueryReaderArrowDatasetIterator
    : public BigQueryReaderDatasetIteratorBase<Dataset> {
 public:
  explicit BigQueryReaderArrowDatasetIterator(
      const typename BigQueryReaderDatasetIteratorBase<Dataset>::Params &params)
      : BigQueryReaderDatasetIteratorBase<Dataset>(params) {
    VLOG(3) << "created BigQueryReaderArrowDatasetIterator for stream: "
            << this->dataset()->stream();
  }

 protected:
  Status EnsureHasRow(bool *end_of_sequence)
      TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) override {
    if (this->response_ && this->response_->has_arrow_record_batch() &&
        this->current_row_index_ <
            this->response_->arrow_record_batch().row_count()) {
      return Status::OK();
    }

    this->response_ = absl::make_unique<apiv1beta1::ReadRowsResponse>();
    if (!this->reader_->Read(this->response_.get())) {
      *end_of_sequence = true;
      return Status::OK();
    }

    this->current_row_index_ = 0;

    auto buffer_ = std::make_shared<arrow::Buffer>(
        reinterpret_cast<const uint8_t *>(&this->response_->arrow_record_batch()
                                               .serialized_record_batch()[0]),
        this->response_->arrow_record_batch().serialized_record_batch().size());

    arrow::io::BufferReader buffer_reader_(buffer_);
    arrow::ipc::DictionaryMemo dict_memo;

    auto arrow_status =
        arrow::ipc::ReadRecordBatch(this->dataset()->arrow_schema(), &dict_memo,
                                    &buffer_reader_, &this->record_batch_);
    if (!arrow_status.ok()) {
      return errors::Internal(arrow_status.ToString());
    }

    VLOG(3) << "got record batch, rows:" << record_batch_->num_rows();

    return Status::OK();
  }

  Status ReadRecord(IteratorContext *ctx, std::vector<Tensor> *out_tensors,
                    const std::vector<string> &columns,
                    const std::vector<DataType> &output_types)
      TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) override {
    out_tensors->clear();

    // Assign Tensors for each column in the current row
    for (size_t i = 0; i < columns.size(); ++i) {
      DataType output_type = output_types[i];
      std::shared_ptr<arrow::Array> arr = this->record_batch_->column(i);

      if (this->current_row_index_ == 0) {
        // Array structure is not going to change, so it is sufficient to check
        // it once.
        TF_RETURN_IF_ERROR(ArrowUtil::CheckArrayType(arr->type(), output_type));
      }

      // Allocate a new tensor and assign Arrow data to it
      Tensor tensor(ctx->allocator({}), output_type, {});
      TF_RETURN_IF_ERROR(
          ArrowUtil::AssignTensor(arr, this->current_row_index_, &tensor));

      out_tensors->emplace_back(std::move(tensor));
    }

    return Status::OK();
  }

 private:
  std::shared_ptr<arrow::RecordBatch> record_batch_ TF_GUARDED_BY(this->mu_);
};

template <typename Dataset>
class BigQueryReaderAvroDatasetIterator
    : public BigQueryReaderDatasetIteratorBase<Dataset> {
 public:
  explicit BigQueryReaderAvroDatasetIterator(
      const typename BigQueryReaderDatasetIteratorBase<Dataset>::Params &params)
      : BigQueryReaderDatasetIteratorBase<Dataset>(params) {
    VLOG(3) << "created BigQueryReaderAvroDatasetIterator for stream: "
            << this->dataset()->stream();
  }

 protected:
  Status EnsureHasRow(bool *end_of_sequence)
      TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) override {
    if (this->response_ &&
        this->current_row_index_ < this->response_->avro_rows().row_count()) {
      return Status::OK();
    }

    this->response_ = absl::make_unique<apiv1beta1::ReadRowsResponse>();
    VLOG(3) << "calling read";
    if (!this->reader_->Read(this->response_.get())) {
      VLOG(3) << "no data";
      *end_of_sequence = true;
      return Status::OK();
    }
    this->current_row_index_ = 0;
    this->decoder_ = avro::binaryDecoder();
    memory_input_stream_ = avro::memoryInputStream(
        reinterpret_cast<const uint8_t *>(
            &this->response_->avro_rows().serialized_binary_rows()[0]),
        this->response_->avro_rows().serialized_binary_rows().size());
    this->decoder_->init(*memory_input_stream_);
    return Status::OK();
  }

  Status ReadRecord(IteratorContext *ctx, std::vector<Tensor> *out_tensors,
                    const std::vector<string> &columns,
                    const std::vector<DataType> &output_types)
      TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) override {
    avro::GenericDatum datum =
        avro::GenericDatum(*this->dataset()->avro_schema());
    avro::decode(*this->decoder_, datum);
    if (datum.type() != avro::AVRO_RECORD) {
      return errors::Unknown("record is not of AVRO_RECORD type");
    }
    const avro::GenericRecord &record = datum.value<avro::GenericRecord>();
    out_tensors->clear();
    // Let's allocate enough space for Tensor, if more than read then slice.
    std::vector<DataType> expected_output_types;
    expected_output_types.reserve(output_types.size());
    for (size_t i = 0; i < columns.size(); i++) {
      const string &column = columns[i];
      const avro::GenericDatum &field = record.field(column);
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
        case avro::AVRO_ARRAY: {
          auto values_vector = field.value<avro::GenericArray>().value();
          if (values_vector.empty())
            dtype = output_types[i];
          else {
            auto value_type = values_vector[0].type();
            if (value_type == avro::AVRO_BOOL)
              dtype = DT_BOOL;
            else if (value_type == avro::AVRO_INT)
              dtype = DT_INT32;
            else if (value_type == avro::AVRO_LONG)
              dtype = DT_INT64;
            else if (value_type == avro::AVRO_FLOAT)
              dtype = DT_FLOAT;
            else if (value_type == avro::AVRO_DOUBLE)
              dtype = DT_DOUBLE;
            else if (value_type == avro::AVRO_STRING)
              dtype = DT_STRING;
            else
              return errors::InvalidArgument(
                  "unsupported data type within AVRO_ARRAY ", value_type);
          }
        } break;
        case avro::AVRO_NULL:
          dtype = output_types[i];
          break;
        default:
          return errors::InvalidArgument("unsupported data type: ",
                                         field.type());
      }
      if (dtype != output_types[i]) {
        return errors::InvalidArgument(
            "output type mismatch for column: ", columns[i],
            " expected type: ", DataType_Name(dtype),
            " actual type: ", DataType_Name(output_types[i]));
      }
      expected_output_types.emplace_back(dtype);
      Tensor tensor(ctx->allocator({}), output_types[i], {});
      out_tensors->emplace_back(std::move(tensor));
    }
    for (size_t i = 0; i < columns.size(); i++) {
      const string &column = columns[i];
      const avro::GenericDatum &field = record.field(column);
      switch (field.type()) {
        case avro::AVRO_BOOL:
          ((*out_tensors)[i]).scalar<bool>()() = field.value<bool>();
          break;
        case avro::AVRO_INT:
          ((*out_tensors)[i]).scalar<int32>()() = field.value<int32_t>();
          break;
        case avro::AVRO_LONG:
          ((*out_tensors)[i]).scalar<int64>()() = field.value<int64_t>();
          break;
        case avro::AVRO_FLOAT:
          ((*out_tensors)[i]).scalar<float>()() = field.value<float>();
          break;
        case avro::AVRO_DOUBLE:
          ((*out_tensors)[i]).scalar<double>()() = field.value<double>();
          break;
        case avro::AVRO_STRING:
          ((*out_tensors)[i]).scalar<tstring>()() = field.value<string>();
          break;
        case avro::AVRO_ENUM:
          ((*out_tensors)[i]).scalar<tstring>()() =
              field.value<avro::GenericEnum>().symbol();
          break;
        case avro::AVRO_BYTES: {
          const std::vector<uint8_t> &field_value =
              field.value<std::vector<uint8_t>>();
          ((*out_tensors)[i]).scalar<tstring>()() =
              string((char *)&field_value[0], field_value.size());
        } break;
        case avro::AVRO_ARRAY: {
          if (output_types[i] == DT_BOOL) {
            auto values_vector = field.value<avro::GenericArray>().value();
            unsigned int size = values_vector.size();
            Tensor output_tensor(ctx->allocator({}), DT_BOOL, {size});
            auto output_flat = output_tensor.flat<bool>();
            for (unsigned int idx = 0; idx < size; idx++) {
              output_flat(idx) = values_vector[idx].value<bool>();
            }
            (*out_tensors)[i] = output_tensor;
          } else if (output_types[i] == DT_INT32) {
            auto values_vector = field.value<avro::GenericArray>().value();
            unsigned int size = values_vector.size();
            Tensor output_tensor(ctx->allocator({}), DT_INT32, {size});
            auto output_flat = output_tensor.flat<int32>();
            for (unsigned int idx = 0; idx < size; idx++) {
              output_flat(idx) = values_vector[idx].value<int32_t>();
            }
            (*out_tensors)[i] = output_tensor;
          } else if (output_types[i] == DT_INT64) {
            auto values_vector = field.value<avro::GenericArray>().value();
            unsigned int size = values_vector.size();
            Tensor output_tensor(ctx->allocator({}), DT_INT64, {size});
            auto output_flat = output_tensor.flat<int64>();
            for (unsigned int idx = 0; idx < size; idx++) {
              output_flat(idx) = values_vector[idx].value<int64_t>();
            }
            (*out_tensors)[i] = output_tensor;
          } else if (output_types[i] == DT_FLOAT) {
            auto values_vector = field.value<avro::GenericArray>().value();
            unsigned int size = values_vector.size();
            Tensor output_tensor(ctx->allocator({}), DT_FLOAT, {size});
            auto output_flat = output_tensor.flat<float>();
            for (unsigned int idx = 0; idx < size; idx++) {
              output_flat(idx) = values_vector[idx].value<float>();
            }
            (*out_tensors)[i] = output_tensor;
          } else if (output_types[i] == DT_DOUBLE) {
            auto values_vector = field.value<avro::GenericArray>().value();
            unsigned int size = values_vector.size();
            Tensor output_tensor(ctx->allocator({}), DT_DOUBLE, {size});
            auto output_flat = output_tensor.flat<double>();
            for (unsigned int idx = 0; idx < size; idx++) {
              output_flat(idx) = values_vector[idx].value<double>();
            }
            (*out_tensors)[i] = output_tensor;
          } else if (output_types[i] == DT_STRING) {
            auto values_vector = field.value<avro::GenericArray>().value();
            unsigned int size = values_vector.size();
            Tensor output_tensor(ctx->allocator({}), DT_STRING, {size});
            auto output_flat = output_tensor.flat<tstring>();
            for (unsigned int idx = 0; idx < size; idx++) {
              output_flat(idx) = values_vector[idx].value<string>();
            }
            (*out_tensors)[i] = output_tensor;
          }
        } break;
        case avro::AVRO_NULL:
          switch (output_types[i]) {
            case DT_BOOL:
              ((*out_tensors)[i]).scalar<bool>()() = false;
              break;
            case DT_INT32:
              ((*out_tensors)[i]).scalar<int32>()() = 0;
              break;
            case DT_INT64:
              ((*out_tensors)[i]).scalar<int64>()() = 0l;
              break;
            case DT_FLOAT:
              ((*out_tensors)[i]).scalar<float>()() = 0.0f;
              break;
            case DT_DOUBLE:
              ((*out_tensors)[i]).scalar<double>()() = 0.0;
              break;
            case DT_STRING:
              ((*out_tensors)[i]).scalar<tstring>()() = "";
              break;
            default:
              return errors::InvalidArgument(
                  "unsupported data type against AVRO_NULL: ", output_types[i]);
          }
          break;
        default:
          return errors::InvalidArgument("unsupported data type: ",
                                         field.type());
      }
    }

    return Status::OK();
  }

 private:
  std::unique_ptr<avro::InputStream> memory_input_stream_
      TF_GUARDED_BY(this->mu_);
  avro::DecoderPtr decoder_ TF_GUARDED_BY(this->mu_);
};

}  // namespace data
}  // namespace tensorflow
#endif  // TENSORFLOW_IO_BIGQUERY_KERNELS_BIGQUERY_LIB_H_
