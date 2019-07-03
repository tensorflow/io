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

#ifndef TENSORFLOW_IO_BIGQUERY_KERNELS_TEST_KERNELS_BIGTABLE_FAKE_H_
#define TENSORFLOW_IO_BIGQUERY_KERNELS_TEST_KERNELS_BIGTABLE_FAKE_H_

#include <string>
#include <vector>

#include "google/cloud/bigquery/storage/v1beta1/storage.grpc.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace apiv1beta1 = ::google::cloud::bigquery::storage::v1beta1;

class FakeBigQueryStorageService
    : public apiv1beta1::BigQueryStorage::BigQueryStorage::Service {
 public:
  string const& project_id() const { return project_id_; }
  string const& table_id() const { return table_id_; }
  string const& dataset_id() const { return dataset_id_; }
  ~FakeBigQueryStorageService() override;

  ::grpc::Status CreateReadSession(
      ::grpc::ServerContext* context,
      const apiv1beta1::CreateReadSessionRequest* request,
      apiv1beta1::ReadSession* response) override;

  ::grpc::Status ReadRows(
      ::grpc::ServerContext* context,
      const apiv1beta1::ReadRowsRequest* request,
      ::grpc::ServerWriter<apiv1beta1::ReadRowsResponse>* writer) override;
  ::grpc::Status BatchCreateReadSessionStreams(
      ::grpc::ServerContext* context,
      const apiv1beta1::BatchCreateReadSessionStreamsRequest* request,
      apiv1beta1::BatchCreateReadSessionStreamsResponse* response) override;
  ::grpc::Status FinalizeStream(
      ::grpc::ServerContext* context,
      const apiv1beta1::FinalizeStreamRequest* request,
      ::google::protobuf::Empty* response) override;
  ::grpc::Status SplitReadStream(
      ::grpc::ServerContext* context,
      const apiv1beta1::SplitReadStreamRequest* request,
      apiv1beta1::SplitReadStreamResponse* response) override;

  void SetAvroSchema(const string& avro_schema) { avro_schema_ = avro_schema; }

  void SetAvroRowsPerStream(
      std::vector<string> avro_serialized_rows_per_stream,
      std::vector<int> avro_serialized_rows_count_per_stream) {
    avro_serialized_rows_per_stream_ = avro_serialized_rows_per_stream;
    avro_serialized_rows_count_per_stream_ =
        avro_serialized_rows_count_per_stream;
  }

  static string BuildStreamName(const string& project_id,
                                const string& table_id,
                                const string& dataset_id, int index) {
    return absl::StrCat(project_id, "/", table_id, "/", dataset_id, "/",
                        std::to_string(index));
  }

 private:
  string project_id_;
  string table_id_;
  string dataset_id_;
  string avro_schema_;
  std::vector<string> streams_;
  std::vector<string> avro_serialized_rows_per_stream_;
  std::vector<int> avro_serialized_rows_count_per_stream_;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_IO_BIGQUERY_KERNELS_TEST_KERNELS_BIGTABLE_FAKE_H_
