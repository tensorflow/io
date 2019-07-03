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
#include "google/cloud/bigquery/storage/v1beta1/storage.grpc.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

Status GrpcStatusToTfStatus(const ::grpc::Status& status);
string GrpcStatusToString(const ::grpc::Status& status);

namespace apiv1beta1 = ::google::cloud::bigquery::storage::v1beta1;

class BigQueryClientResource : public ResourceBase {
 public:
  explicit BigQueryClientResource(
      std::shared_ptr<apiv1beta1::BigQueryStorage::Stub> stub)
      : stub_(std::move(stub)) {}

  std::shared_ptr<apiv1beta1::BigQueryStorage::Stub> get_stub() {
    return stub_;
  }

  string DebugString() const override {
    return "BigQueryClientResource";
  }

 private:
  std::shared_ptr<apiv1beta1::BigQueryStorage::Stub> stub_;
};

namespace data {

// BigQueryReaderDatasetIterator is an abstract class for iterators from
// datasets that are "readers" (source datasets, not transformation datasets)
// that read from BigQuery.
template <typename Dataset>
class BigQueryReaderDatasetIterator : public DatasetIterator<Dataset> {
 public:
  explicit BigQueryReaderDatasetIterator(
      const typename DatasetIterator<Dataset>::Params& params)
      : DatasetIterator<Dataset>(params) {
    VLOG(3) << "created BigQueryReaderDatasetIterator for stream: "
            << this->dataset()->stream();
  }

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    mutex_lock l(mu_);
    VLOG(3) << "calling BigQueryReaderDatasetIterator.GetNextInternal() index: "
            << current_row_index_
            << " stream: "
            << this->dataset()->stream();
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

 private:
  Status EnsureReaderInitialized() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (reader_) {
      return Status::OK();
    }

    apiv1beta1::ReadRowsRequest readRowsRequest;
    readRowsRequest.mutable_read_position()->mutable_stream()->set_name(
        this->dataset()->stream());
    readRowsRequest.mutable_read_position()->set_offset(0);

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

  Status EnsureHasRow(bool* end_of_sequence) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (response_ && current_row_index_ < response_->avro_rows().row_count()) {
      return Status::OK();
    }

    response_ = absl::make_unique<apiv1beta1::ReadRowsResponse>();
    VLOG(3) << "calling read";
    if (!reader_->Read(response_.get())) {
      VLOG(3) << "no data";
      *end_of_sequence = true;
      return Status::OK();
    }
    current_row_index_ = 0;
    decoder_ = avro::binaryDecoder();
    memory_input_stream_ = avro::memoryInputStream(
        reinterpret_cast<const uint8_t*>(
            &response_->avro_rows().serialized_binary_rows()[0]),
        response_->avro_rows().serialized_binary_rows().size());
    decoder_->init(*memory_input_stream_);
    return Status::OK();
  }

  Status ReadRecord(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                    const std::vector<string>& columns,
                    const std::vector<DataType>& output_types)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    avro::GenericDatum datum = avro::GenericDatum(*this->dataset()->schema());
    avro::decode(*decoder_, datum);
    if (datum.type() != avro::AVRO_RECORD) {
      return errors::Unknown("record is not of AVRO_RECORD type");
    }
    const avro::GenericRecord& record = datum.value<avro::GenericRecord>();
    out_tensors->clear();
    // Let's allocate enough space for Tensor, if more than read then slice.
    std::vector<DataType> expected_output_types;
    expected_output_types.reserve(output_types.size());
    for (size_t i = 0; i < columns.size(); i++) {
      const string& column = columns[i];
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
          return errors::InvalidArgument("unsupported data type: ",
                                         field.type());
      }
      if (dtype != output_types[i] && output_types[i] != DT_STRING) {
        return errors::InvalidArgument(
            "output type mismatch for column: ", columns[i],
            " expected type: ", DataType_Name(dtype),
            " actual type: ", DataType_Name(output_types[i]));
        return errors::InvalidArgument("error");
      }
      expected_output_types.emplace_back(dtype);
      Tensor tensor(ctx->allocator({}), output_types[i], {});
      out_tensors->emplace_back(std::move(tensor));
    }
    for (size_t i = 0; i < columns.size(); i++) {
      const string& column = columns[i];
      const avro::GenericDatum& field = record.field(column);
      switch (field.type()) {
        case avro::AVRO_BOOL:
          if (output_types[i] == DT_BOOL) {
            ((*out_tensors)[i]).scalar<bool>()() = field.value<bool>();
          } else if (output_types[i] == DT_STRING) {
            ((*out_tensors)[i]).scalar<string>()() =
                std::to_string(field.value<bool>());
          }
          break;
        case avro::AVRO_INT:
          if (output_types[i] == DT_INT32) {
            ((*out_tensors)[i]).scalar<int32>()() = field.value<int32_t>();
          } else if (output_types[i] == DT_STRING) {
            ((*out_tensors)[i]).scalar<string>()() =
                std::to_string(field.value<int32_t>());
          }
          break;
        case avro::AVRO_LONG:
          if (output_types[i] == DT_INT64) {
            ((*out_tensors)[i]).scalar<int64>()() = field.value<int64_t>();
          } else if (output_types[i] == DT_STRING) {
            ((*out_tensors)[i]).scalar<string>()() =
                std::to_string(field.value<int64_t>());
          }
          break;
        case avro::AVRO_FLOAT:
          if (output_types[i] == DT_FLOAT) {
            ((*out_tensors)[i]).scalar<float>()() = field.value<float>();
          } else if (output_types[i] == DT_STRING) {
            ((*out_tensors)[i]).scalar<string>()() =
                std::to_string(field.value<float>());
          }
          break;
        case avro::AVRO_DOUBLE:
          if (output_types[i] == DT_DOUBLE) {
            ((*out_tensors)[i]).scalar<double>()() = field.value<double>();
          } else if (output_types[i] == DT_STRING) {
            ((*out_tensors)[i]).scalar<string>()() =
                std::to_string(field.value<double>());
          }
          break;
        case avro::AVRO_STRING:
          ((*out_tensors)[i]).scalar<string>()() = field.value<string>();
          break;
        case avro::AVRO_ENUM:
          ((*out_tensors)[i]).scalar<string>()() =
              field.value<avro::GenericEnum>().symbol();
          break;
        default:
          return errors::InvalidArgument("unsupported data type: ",
                                         field.type());
      }
    }

    return Status::OK();
  }

  int current_row_index_ = 0;
  mutex mu_;
  std::unique_ptr<::grpc::ClientContext> read_rows_context_ GUARDED_BY(mu_);
  std::unique_ptr<::grpc::ClientReader<apiv1beta1::ReadRowsResponse>> reader_
      GUARDED_BY(mu_);
  std::unique_ptr<apiv1beta1::ReadRowsResponse> response_ GUARDED_BY(mu_);
  std::unique_ptr<avro::InputStream> memory_input_stream_ GUARDED_BY(mu_);
  avro::DecoderPtr decoder_ GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow
#endif  // TENSORFLOW_IO_BIGQUERY_KERNELS_BIGQUERY_LIB_H_
