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

#include <grpc++/grpc++.h>
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
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/net.h"
#include "tensorflow_io/bigquery/kernels/bigquery_lib.h"

namespace tensorflow {
namespace {

namespace apiv1beta1 = ::google::cloud::bigquery::storage::v1beta1;

class BigQueryTestClientOp : public OpKernel {
 public:
  explicit BigQueryTestClientOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fake_server_address", &fake_server_address_));
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

      BigQueryClientResource* resource;
      OP_REQUIRES_OK(
          ctx,
          mgr->LookupOrCreate<BigQueryClientResource>(
              cinfo_.container(), cinfo_.name(), &resource,
              [this](BigQueryClientResource** ret) EXCLUSIVE_LOCKS_REQUIRED(
                  mu_) {
                LOG(INFO) << "Connecting BigQueryTestClientOp Fake client to:"
                          << fake_server_address_;
                std::shared_ptr<grpc::Channel> channel = ::grpc::CreateChannel(
                    fake_server_address_, grpc::InsecureChannelCredentials());
                auto stub = apiv1beta1::BigQueryStorage::NewStub(channel);
                LOG(INFO) << "BigQueryTestClientOp waiting for connections";
                channel->WaitForConnected(
                  gpr_time_add(gpr_now(GPR_CLOCK_REALTIME),
                  gpr_time_from_seconds(15, GPR_TIMESPAN)));
                LOG(INFO) << "Done creating BigQueryTestClientOp Fake client";
                *ret = new BigQueryClientResource(std::move(stub));
                return Status::OK();
              }));

      core::ScopedUnref unref_resource(resource);
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
  string fake_server_address_;
};

REGISTER_KERNEL_BUILDER(Name("BigQueryTestClient").Device(DEVICE_CPU),
                        BigQueryTestClientOp);

}  // namespace
}  // namespace tensorflow
