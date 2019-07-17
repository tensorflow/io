/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_io/bigquery/kernels/test_kernels/fake_bigquery_storage_service.h"

namespace tensorflow {

namespace apiv1beta1 = ::google::cloud::bigquery::storage::v1beta1;

::grpc::Status FakeBigQueryStorageService::CreateReadSession(
    ::grpc::ServerContext* context,
    const apiv1beta1::CreateReadSessionRequest* request,
    apiv1beta1::ReadSession* response) {
  CHECK(context);
  CHECK(request);
  CHECK(response);
  project_id_ = request->table_reference().project_id();
  this->table_id_ = request->table_reference().table_id();
  this->dataset_id_ = request->table_reference().dataset_id();
  this->streams_.clear();
  response->mutable_streams()->Clear();
  for (int i = 0; i < request->requested_streams(); i++) {
    string stream_name = BuildStreamName(this->project_id_, this->table_id_,
                                         this->dataset_id_, i);
    this->streams_.emplace_back(stream_name);
    apiv1beta1::Stream* stream = response->mutable_streams()->Add();
    stream->set_name(stream_name);
  }
  response->mutable_avro_schema()->set_schema(avro_schema_);

  return ::grpc::Status::OK;
}

::grpc::Status FakeBigQueryStorageService::ReadRows(
    ::grpc::ServerContext* context, const apiv1beta1::ReadRowsRequest* request,
    ::grpc::ServerWriter<apiv1beta1::ReadRowsResponse>* writer) {
  auto it = std::find(streams_.begin(), streams_.end(),
                      request->read_position().stream().name());
  if (it != streams_.end()) {
    uint index = std::distance(streams_.begin(), it);
    if (index < avro_serialized_rows_per_stream_.size()) {
      apiv1beta1::ReadRowsResponse response;
      response.mutable_avro_rows()->set_serialized_binary_rows(
          avro_serialized_rows_per_stream_.at(index));
      response.mutable_avro_rows()->set_row_count(
          avro_serialized_rows_count_per_stream_.at(index));
      writer->Write(response);
    }
  } else {
    ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                   absl::StrCat("Stream not found:",
                                request->read_position().stream().name()));
  }

  return ::grpc::Status::OK;
}

::grpc::Status FakeBigQueryStorageService::BatchCreateReadSessionStreams(
    ::grpc::ServerContext* context,
    const apiv1beta1::BatchCreateReadSessionStreamsRequest* request,
    apiv1beta1::BatchCreateReadSessionStreamsResponse* response) {
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Not implemented.");
}

::grpc::Status FakeBigQueryStorageService::FinalizeStream(
    ::grpc::ServerContext* context,
    const apiv1beta1::FinalizeStreamRequest* request,
    ::google::protobuf::Empty* response) {
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Not implemented.");
}

::grpc::Status FakeBigQueryStorageService::SplitReadStream(
    ::grpc::ServerContext* context,
    const apiv1beta1::SplitReadStreamRequest* request,
    apiv1beta1::SplitReadStreamResponse* response) {
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Not implemented.");
}

FakeBigQueryStorageService::~FakeBigQueryStorageService() {}

}  // namespace tensorflow
