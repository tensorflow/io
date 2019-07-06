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

#include <grpcpp/grpcpp.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "api/Compiler.hh"
#include "api/DataFile.hh"
#include "api/Decoder.hh"
#include "api/Encoder.hh"
#include "api/Generic.hh"
#include "api/Specific.hh"
#include "api/ValidSchema.hh"
#include "google/cloud/bigquery/storage/v1beta1/storage.grpc.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/net.h"
#include "tensorflow_io/bigquery/kernels/bigquery_lib.h"
#include "tensorflow_io/bigquery/kernels/test_kernels/fake_bigquery_storage_service.h"

namespace tensorflow {
namespace {

namespace apiv1beta1 = ::google::cloud::bigquery::storage::v1beta1;

class BigQueryTestClientResource : public BigQueryClientResource {
 public:
  explicit BigQueryTestClientResource(
      std::shared_ptr<apiv1beta1::BigQueryStorage::Stub> stub,
      std::unique_ptr<grpc::Server> server,
      std::unique_ptr<FakeBigQueryStorageService> fake_service)
      : BigQueryClientResource(stub),
        server_(std::move(server)),
        fake_service_(std::move(fake_service)) {}

  string DebugString() const override { return "BigQueryTestClientResource"; }

 protected:
  ~BigQueryTestClientResource() override { server_->Shutdown(); }

 private:
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<FakeBigQueryStorageService> fake_service_;
};

class BigQueryTestClientOp : public OpKernel {
 public:
  explicit BigQueryTestClientOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("avro_schema", &avro_schema_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("avro_serialized_rows_per_stream",
                                     &avro_serialized_rows_per_stream_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("avro_serialized_rows_count_per_stream",
                                     &avro_serialized_rows_count_per_stream_));
    OP_REQUIRES(
        ctx,
        avro_serialized_rows_per_stream_.size() ==
            avro_serialized_rows_count_per_stream_.size(),
        errors::InvalidArgument("avro_serialized_rows_per_stream and "
                                "avro_serialized_rows_count_per_stream lists "
                                "should have same length"));
  }

  ~BigQueryTestClientOp() override {
    if (cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->Delete<BigQueryClientResource>(cinfo_.container(),
                                                cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

  void Compute(OpKernelContext* ctx) override LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (!initialized_) {
      ResourceMgr* mgr = ctx->resource_manager();
      OP_REQUIRES_OK(ctx, cinfo_.Init(mgr, def()));

      BigQueryTestClientResource* test_resource;
      OP_REQUIRES_OK(
          ctx,
          mgr->LookupOrCreate<BigQueryTestClientResource>(
              cinfo_.container(), cinfo_.name(), &test_resource,
              [this](BigQueryTestClientResource** ret) EXCLUSIVE_LOCKS_REQUIRED(
                  mu_) {
                auto fake_service =
                    absl::make_unique<FakeBigQueryStorageService>();
                fake_service->SetAvroSchema(this->avro_schema_);
                fake_service->SetAvroRowsPerStream(
                    this->avro_serialized_rows_per_stream_,
                    this->avro_serialized_rows_count_per_stream_);

                grpc::ServerBuilder server_builder;
                int port = internal::PickUnusedPortOrDie();
                string server_address = absl::StrCat("localhost:", port);
                server_builder.AddListeningPort(
                    server_address, ::grpc::InsecureServerCredentials(), &port);
                server_builder.RegisterService(fake_service.get());
                std::unique_ptr<grpc::Server> server =
                    server_builder.BuildAndStart();

                string address =
                    absl::StrCat("localhost:", port);
                std::shared_ptr<grpc::Channel> channel = ::grpc::CreateChannel(
                    address, grpc::InsecureChannelCredentials());
                auto stub =
                    absl::make_unique<apiv1beta1::BigQueryStorage::Stub>(
                        channel);

                channel->WaitForConnected(
                  gpr_time_add(gpr_now(GPR_CLOCK_REALTIME),
                  gpr_time_from_seconds(5, GPR_TIMESPAN)));
                VLOG(3) << "Done creating Fake client";
                *ret = new BigQueryTestClientResource(std::move(stub),
                                                      std::move(server),
                                                      std::move(fake_service));
                return Status::OK();
              }));

      BigQueryClientResource* resource;
      OP_REQUIRES_OK(ctx, mgr->LookupOrCreate<BigQueryClientResource>(
                              cinfo_.container(), cinfo_.name(), &resource,
                              [test_resource](BigQueryClientResource** ret)
                                  EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                                    *ret = new BigQueryClientResource(
                                        test_resource->get_stub());
                                    return Status::OK();
                                  }));
      core::ScopedUnref unref_resource(resource);
      core::ScopedUnref unref_test_resource(test_resource);
      initialized_ = true;
    }
    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo_.container(), cinfo_.name(),
                            MakeTypeIndex<BigQueryClientResource>()));
  }

 private:
  mutex mu_;
  ContainerInfo cinfo_ GUARDED_BY(mu_);
  bool initialized_ GUARDED_BY(mu_) = false;
  string avro_schema_;
  std::vector<string> avro_serialized_rows_per_stream_;
  std::vector<int> avro_serialized_rows_count_per_stream_;
};

REGISTER_KERNEL_BUILDER(Name("BigQueryTestClient").Device(DEVICE_CPU),
                        BigQueryTestClientOp);

}  // namespace
}  // namespace tensorflow
