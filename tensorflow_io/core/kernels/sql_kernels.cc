/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

#include <libpq-fe.h>

namespace tensorflow {
namespace io {
namespace {

class SqlIterableResource : public ResourceBase {
 public:
  SqlIterableResource(Env* env) : env_(env) {}
  ~SqlIterableResource() {}

  Status Init(const string& topic, const int32 partition, const int64 offset,
              const std::vector<string>& metadata) {
    mutex_lock l(mu_);


  const char conninfo[] =
      "postgresql://postgres@localhost?port=5432&dbname=libpq_demo";
  PGconn *conn = PQconnectdb(conninfo);
  /* Check to see that the backend connection was successfully made */
  if (PQstatus(conn) != CONNECTION_OK) {
    std::cout << "Connection to database failed: " << PQerrorMessage(conn)
              << std::endl;
    PQfinish(conn);
    exit(1);
  } else {
    std::cout << "Connection to database succeed." << std::endl;
  }

  PGresult *res = NULL;

  /* create table demo */
  res = PQexec(conn, "create table if not exists t(id int, name text);");
  if (PQresultStatus(res) != PGRES_COMMAND_OK) {
    std::cout << "Create table failed: " << PQresultErrorMessage(res)
              << std::endl;
    PQclear(res);
    exit(1);
  }
  PQclear(res);

    return Status::OK();
  }
  Status Read(const int64 index,
              std::function<Status(const TensorShape& shape, Tensor** message,
                                   Tensor** key)>
                  allocate_func) {
    mutex_lock l(mu_);
    return Status::OK();
  }
  string DebugString() const override { return "SqlIterableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class SqlIterableInitOp : public ResourceOpKernel<SqlIterableResource> {
 public:
  explicit SqlIterableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<SqlIterableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<SqlIterableResource>::Compute(context);

    const Tensor* topic_tensor;
    OP_REQUIRES_OK(context, context->input("topic", &topic_tensor));
    const string& topic = topic_tensor->scalar<string>()();

    const Tensor* partition_tensor;
    OP_REQUIRES_OK(context, context->input("partition", &partition_tensor));
    const int32 partition = partition_tensor->scalar<int32>()();

    const Tensor* offset_tensor;
    OP_REQUIRES_OK(context, context->input("offset", &offset_tensor));
    const int64 offset = offset_tensor->scalar<int64>()();

    const Tensor* metadata_tensor;
    OP_REQUIRES_OK(context, context->input("metadata", &metadata_tensor));
    std::vector<string> metadata;
    for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
      metadata.push_back(metadata_tensor->flat<string>()(i));
    }

    OP_REQUIRES_OK(context,
                   resource_->Init(topic, partition, offset, metadata));
  }
  Status CreateResource(SqlIterableResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new SqlIterableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class SqlIterableReadOp : public OpKernel {
 public:
  explicit SqlIterableReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    SqlIterableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* index_tensor;
    OP_REQUIRES_OK(context, context->input("index", &index_tensor));
    const int64 index = index_tensor->scalar<int64>()();

    OP_REQUIRES_OK(
        context,
        resource->Read(
            index,
            [&](const TensorShape& shape, Tensor** message,
                Tensor** key) -> Status {
              TF_RETURN_IF_ERROR(context->allocate_output(0, shape, message));
              TF_RETURN_IF_ERROR(context->allocate_output(1, shape, key));
              return Status::OK();
            }));
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>SqlIterableInit").Device(DEVICE_CPU),
                        SqlIterableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>SqlIterableRead").Device(DEVICE_CPU),
                        SqlIterableReadOp);
}  // namespace
}  // namespace io
}  // namespace tensorflow
