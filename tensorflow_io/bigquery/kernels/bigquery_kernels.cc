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
#include <string>

#include "api/Compiler.hh"
#include "api/DataFile.hh"
#include "api/Decoder.hh"
#include "api/Encoder.hh"
#include "api/Generic.hh"
#include "api/Specific.hh"
#include "api/ValidSchema.hh"
#include "google/cloud/bigquery/storage/v1beta1/storage.grpc.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow_io/bigquery/kernels/bigquery_lib.h"

namespace tensorflow {
namespace {

namespace apiv1beta1 = ::google::cloud::bigquery::storage::v1beta1;
static constexpr int kMaxReceiveMessageSize = 1 << 24;  // 16 MBytes

class BigQueryClientOp : public OpKernel {
 public:
  explicit BigQueryClientOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  ~BigQueryClientOp() override {
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
      BigQueryClientResource* resource;
      OP_REQUIRES_OK(
          ctx,
          mgr->LookupOrCreate<BigQueryClientResource>(
              cinfo_.container(), cinfo_.name(), &resource,
              [this, ctx](
                  BigQueryClientResource** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                std::string server_name =
                    "dns:///bigquerystorage.googleapis.com";
                auto creds = ::grpc::GoogleDefaultCredentials();
                grpc::ChannelArguments args;
                args.SetMaxReceiveMessageSize(kMaxReceiveMessageSize);
                args.SetUserAgentPrefix("tensorflow");
                args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 0);
                args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 60 * 1000);
                auto channel = ::grpc::CreateCustomChannel(
                    server_name, creds, args);
                VLOG(3) << "Creating GRPC channel";
                auto stub =
                    absl::make_unique<apiv1beta1::BigQueryStorage::Stub>(
                        channel);
                VLOG(3) << "Done creating GRPC channel";

                *ret = new BigQueryClientResource(std::move(stub));
                return Status::OK();
              }));
      core::ScopedUnref resource_cleanup(resource);
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
};

REGISTER_KERNEL_BUILDER(Name("IO>BigQueryClient").Device(DEVICE_CPU),
                        BigQueryClientOp);

class BigQueryReadSessionOp : public OpKernel {
 public:
  explicit BigQueryReadSessionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("parent", &parent_));
    OP_REQUIRES(ctx, !parent_.empty(),
                errors::InvalidArgument("parent must be non-empty"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("project_id", &project_id_));
    OP_REQUIRES(ctx, !project_id_.empty(),
                errors::InvalidArgument("project_id must be non-empty"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_id", &table_id_));
    OP_REQUIRES(ctx, !table_id_.empty(),
                errors::InvalidArgument("table_id must be non-empty"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dataset_id", &dataset_id_));
    OP_REQUIRES(ctx, !dataset_id_.empty(),
                errors::InvalidArgument("dataset_id must be non-empty"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("selected_fields", &selected_fields_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("row_restriction", &row_restriction_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("requested_streams", &requested_streams_));
  }

  void Compute(OpKernelContext* ctx) override LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    ResourceMgr* mgr = ctx->resource_manager();
    OP_REQUIRES_OK(ctx, cinfo_.Init(mgr, def()));

    BigQueryClientResource* client_resource;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &client_resource));
    core::ScopedUnref scoped_unref(client_resource);

    apiv1beta1::CreateReadSessionRequest createReadSessionRequest;
    createReadSessionRequest.mutable_table_reference()->set_project_id(
        project_id_);
    createReadSessionRequest.mutable_table_reference()->set_dataset_id(
        dataset_id_);
    createReadSessionRequest.mutable_table_reference()->set_table_id(table_id_);
    createReadSessionRequest.set_parent(parent_);
    *createReadSessionRequest.mutable_read_options()
         ->mutable_selected_fields() = {selected_fields_.begin(),
                                        selected_fields_.end()};
    createReadSessionRequest.mutable_read_options()->set_row_restriction(
        row_restriction_);                                        
    createReadSessionRequest.set_requested_streams(requested_streams_);
    createReadSessionRequest.set_sharding_strategy(apiv1beta1::ShardingStrategy::BALANCED);
    createReadSessionRequest.set_format(apiv1beta1::DataFormat::AVRO);
    VLOG(3) << "createReadSessionRequest: "
            << createReadSessionRequest.DebugString();
    ::grpc::ClientContext context;
    context.AddMetadata("x-goog-request-params",
                        strings::Printf("table_reference.dataset_id=%s&table_"
                                        "reference.project_id=%s",
                                        dataset_id_.c_str(),
                                        project_id_.c_str()));
    context.set_deadline(gpr_time_add(gpr_now(GPR_CLOCK_REALTIME),
                         gpr_time_from_seconds(60, GPR_TIMESPAN)));

    std::shared_ptr<apiv1beta1::ReadSession> readSessionResponse =
        std::make_shared<apiv1beta1::ReadSession>();
    VLOG(3) << "calling readSession";
    ::grpc::Status status = client_resource->get_stub()->CreateReadSession(
        &context, createReadSessionRequest, readSessionResponse.get());
    if (!status.ok()) {
      VLOG(3) << "readSession status:" << GrpcStatusToString(status);
      ctx->CtxFailure(GrpcStatusToTfStatus(status));
      return;
    }
    VLOG(3) << "readSession response:" << readSessionResponse->DebugString();
    if (readSessionResponse->has_avro_schema()) {
      VLOG(3) << "avro schema:" << readSessionResponse->avro_schema().schema();
    }

    Tensor* streams_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        "streams",
        {readSessionResponse->streams_size()},
        &streams_t));
    auto streams_vec = streams_t->vec<tstring>();
    for (int i = 0; i < readSessionResponse->streams_size(); i++) {
      streams_vec(i) = readSessionResponse->streams(i).name();
    }
    Tensor* avro_schema_t = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("avro_schema", {}, &avro_schema_t));
    avro_schema_t->scalar<tstring>()() =
        readSessionResponse->avro_schema().schema();
  }

 private:
  // Note: these fields are const after construction.
  string parent_;
  string project_id_;
  string table_id_;
  string dataset_id_;
  std::vector<string> selected_fields_;
  std::vector<DataType> output_types_;
  string row_restriction_;
  int requested_streams_;

  mutex mu_;
  ContainerInfo cinfo_ GUARDED_BY(mu_);
  bool initialized_ GUARDED_BY(mu_) = false;
};

REGISTER_KERNEL_BUILDER(Name("IO>BigQueryReadSession").Device(DEVICE_CPU),
                        BigQueryReadSessionOp);

}  // namespace
}  // namespace tensorflow
