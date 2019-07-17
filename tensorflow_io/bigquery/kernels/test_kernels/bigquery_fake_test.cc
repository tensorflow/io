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

#include <gmock/gmock.h>
#include <grpcpp/grpcpp.h>

#include "google/cloud/bigquery/storage/v1beta1/storage.grpc.pb.h"
#include "gtest/gtest.h"
#include "tensorflow/core/platform/net.h"
#include "tensorflow_io/bigquery/kernels/test_kernels/fake_bigquery_storage_service.h"

namespace tensorflow {

using ::testing::Eq;
using ::testing::Test;

namespace apiv1beta1 = ::google::cloud::bigquery::storage::v1beta1;

class BigqueryFakeTest : public Test {
 protected:
  BigqueryFakeTest()
      : server_address_(
            absl::StrCat("localhost:", internal::PickUnusedPortOrDie())),
        channel_(::grpc::CreateChannel(server_address_,
                                       grpc::InsecureChannelCredentials())),
        fake_service_(absl::make_unique<FakeBigQueryStorageService>()),
        stub_(absl::make_unique<apiv1beta1::BigQueryStorage::Stub>(
            channel_)) {
    grpc::ServerBuilder server_builder;
    server_builder.AddListeningPort(server_address_,
                                    ::grpc::InsecureServerCredentials());
    server_builder.RegisterService(fake_service_.get());
    server_ = server_builder.BuildAndStart();
  }

  void SetUp() override {
    ASSERT_TRUE(
        channel_->WaitForConnected(gpr_time_add(gpr_now(GPR_CLOCK_REALTIME),
                          gpr_time_from_seconds(5, GPR_TIMESPAN))));
  }

  void TearDown() override { server_->Shutdown(); }

  // The address for clients to connect to.
  const string server_address_;

  // Channels for client connection to the service.
  std::shared_ptr<grpc::Channel> channel_;

  // The actual fake server that is under test.
  std::unique_ptr<FakeBigQueryStorageService> fake_service_;

  // A server on which we will stand up the Reservation service.
  std::unique_ptr<grpc::Server> server_;

  // Service stub to send messages to the service.
  std::unique_ptr<apiv1beta1::BigQueryStorage::Stub> stub_;
};

TEST_F(BigqueryFakeTest, CreateReadSession) {
  fake_service_->SetAvroSchema("some avro schema");
  apiv1beta1::CreateReadSessionRequest createReadSessionRequest;
  createReadSessionRequest.set_requested_streams(3);
  createReadSessionRequest.mutable_table_reference()->set_project_id(
      "test_project_id");
  createReadSessionRequest.mutable_table_reference()->set_table_id(
      "test_table_id");
  createReadSessionRequest.mutable_table_reference()->set_dataset_id(
      "test_dataset_id");
  ::grpc::ClientContext context;
  apiv1beta1::ReadSession readSessionResponse;
  VLOG(3) << "calling readSession";
  ::grpc::Status status = stub_->CreateReadSession(
      &context, createReadSessionRequest, &readSessionResponse);
  VLOG(3) << "readSessionResponse: " << readSessionResponse.DebugString();
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(readSessionResponse.streams_size(),
              Eq(createReadSessionRequest.requested_streams()));
}

TEST_F(BigqueryFakeTest, ReadRows) {
  apiv1beta1::CreateReadSessionRequest createReadSessionRequest;
  createReadSessionRequest.set_requested_streams(3);
  createReadSessionRequest.mutable_table_reference()->set_project_id(
      "test_project_id");
  createReadSessionRequest.mutable_table_reference()->set_table_id(
      "test_table_id");
  createReadSessionRequest.mutable_table_reference()->set_dataset_id(
      "test_dataset_id");
  ::grpc::ClientContext context;
  apiv1beta1::ReadSession readSessionResponse;
  VLOG(3) << "calling readSession";
  ::grpc::Status status = stub_->CreateReadSession(
      &context, createReadSessionRequest, &readSessionResponse);
  VLOG(3) << "readSessionResponse: " << readSessionResponse.DebugString();
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(readSessionResponse.streams_size(),
              Eq(createReadSessionRequest.requested_streams()));
  fake_service_->SetAvroSchema("some avro schema");
  std::vector<string> avro_serialized_rows_per_stream = {"stream0", "stream1"};
  std::vector<int> avro_serialized_rows_count_per_stream = {1, 1};
  fake_service_->SetAvroRowsPerStream(avro_serialized_rows_per_stream,
                                      avro_serialized_rows_count_per_stream);

  string stream_name = FakeBigQueryStorageService::BuildStreamName(
      createReadSessionRequest.table_reference().project_id(),
      createReadSessionRequest.table_reference().table_id(),
      createReadSessionRequest.table_reference().dataset_id(), 0);

  apiv1beta1::ReadRowsRequest readRowsRequest;
  readRowsRequest.mutable_read_position()->mutable_stream()->set_name(
      stream_name);
  readRowsRequest.mutable_read_position()->set_offset(0);

  ::grpc::ClientContext context2;
  auto reader_ = stub_->ReadRows(&context2, readRowsRequest);
  apiv1beta1::ReadRowsResponse response;
  bool result = reader_->Read(&response);
  EXPECT_TRUE(result);
  EXPECT_THAT(response.avro_rows().serialized_binary_rows(), Eq("stream0"));
}
}  // namespace tensorflow
