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

#include <libpq-fe.h>
#include <pg_config_types.h>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {
namespace io {
namespace {

DataType SqlDataType(int oid) {
  switch (oid) {
    case INT4OID:
      return DT_INT32;
    case INT8OID:
      return DT_INT64;
    case FLOAT4OID:
      return DT_FLOAT;
    case FLOAT8OID:
      return DT_DOUBLE;
  }
  return DT_INVALID;
};

Status SqlDataCopy(int oid, char* data, Tensor* value) {
  bool result = false;
  switch (oid) {
    case INT4OID:
      result = absl::SimpleAtoi<int32>(data, &value->flat<int32>()(0));
      break;
    case INT8OID:
      result = absl::SimpleAtoi<int64>(data, &value->flat<int64>()(0));
      break;
    case FLOAT4OID:
      result = absl::SimpleAtof(data, &value->flat<float>()(0));
      break;
    case FLOAT8OID:
      result = absl::SimpleAtod(data, &value->flat<double>()(0));
      break;
    default:
      return errors::InvalidArgument("sql data type ", oid, " not supported");
  }
  if (!result) {
    return errors::InvalidArgument("unable to convert data");
  }
  return Status::OK();
};

class SqlIterableResource : public ResourceBase {
 public:
  SqlIterableResource(Env* env)
      : env_(env),
        conn_(nullptr,
              [](PGconn* p) {
                if (p != nullptr) {
                  PQfinish(p);
                }
              }),
        result_(nullptr, [](PGresult* p) {
          if (p != nullptr) {
            PQclear(p);
          }
        }) {}
  ~SqlIterableResource() {}

  Status Init(const string& query, const string& endpoint, int64* count,
              std::vector<string>* fields, std::vector<DataType>* dtypes) {
    mutex_lock l(mu_);
    conn_.reset(PQconnectdb(endpoint.c_str()));
    if (PQstatus(conn_.get()) != CONNECTION_OK) {
      return errors::InvalidArgument("Connection to database failed: ",
                                     PQerrorMessage(conn_.get()));
    }
    LOG(INFO) << "Connection to database succeed.";

    result_.reset(PQexec(conn_.get(), query.c_str()));
    if (PQresultStatus(result_.get()) != PGRES_TUPLES_OK) {
      return errors::InvalidArgument("Exec of query failed: ",
                                     PQerrorMessage(conn_.get()));
    }
    LOG(INFO) << "Exec of query succeed.";

    count_ = PQntuples(result_.get());
    fields_.clear();
    dtypes_.clear();
    int field_count = PQnfields(result_.get());
    for (int i = 0; i < field_count; i++) {
      const char* field = PQfname(result_.get(), i);
      int oid = PQftype(result_.get(), i);
      DataType dtype = SqlDataType(oid);
      if (dtype == DT_INVALID) {
        return errors::InvalidArgument("OID of data type ", oid,
                                       " is not supported");
      }
      fields_.push_back(field);
      dtypes_.push_back(dtype);
    }

    *count = count_;
    fields->clear();
    fields->reserve(fields_.size());
    for (size_t i = 0; i < fields_.size(); i++) {
      fields->push_back(fields_[i]);
    }
    dtypes->clear();
    dtypes->reserve(dtypes_.size());
    for (size_t i = 0; i < dtypes_.size(); i++) {
      dtypes->push_back(dtypes_[i]);
    }

    return Status::OK();
  }
  Status Read(const int64 index, const Tensor* field_tensor,
              std::function<Status(const int64 field_index, Tensor** value)>
                  allocate_func) {
    mutex_lock l(mu_);
    for (int64 i = 0; i < field_tensor->NumElements(); i++) {
      Tensor* value;
      TF_RETURN_IF_ERROR(allocate_func(i, &value));
      char* field_name = (char*)(field_tensor->flat<tstring>()(i).c_str());
      int field_number = PQfnumber(result_.get(), field_name);
      int field_type = PQftype(result_.get(), field_number);
      char* data = PQgetvalue(result_.get(), index, field_number);
      TF_RETURN_IF_ERROR(SqlDataCopy(field_type, data, value));
    }
    return Status::OK();
  }
  string DebugString() const override { return "SqlIterableResource"; }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<PGconn, void (*)(PGconn*)> conn_ TF_GUARDED_BY(mu_);
  std::unique_ptr<PGresult, void (*)(PGresult*)> result_ TF_GUARDED_BY(mu_);
  int64 count_ TF_GUARDED_BY(mu_);
  std::vector<string> fields_ TF_GUARDED_BY(mu_);
  std::vector<DataType> dtypes_ TF_GUARDED_BY(mu_);
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

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<tstring>()();

    const Tensor* endpoint_tensor;
    OP_REQUIRES_OK(context, context->input("endpoint", &endpoint_tensor));
    const string& endpoint = endpoint_tensor->scalar<tstring>()();

    int64 count;
    std::vector<string> fields;
    std::vector<DataType> dtypes;
    OP_REQUIRES_OK(context,
                   resource_->Init(input, endpoint, &count, &fields, &dtypes));

    Tensor* count_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}), &count_tensor));
    count_tensor->scalar<int64>()() = count;

    Tensor* field_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       2, TensorShape({static_cast<int64>(fields.size())}),
                       &field_tensor));
    for (size_t i = 0; i < fields.size(); i++) {
      field_tensor->flat<tstring>()(i) = fields[i];
    }
    Tensor* dtype_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       3, TensorShape({static_cast<int64>(dtypes.size())}),
                       &dtype_tensor));
    for (size_t i = 0; i < dtypes.size(); i++) {
      dtype_tensor->flat<int64>()(i) = dtypes[i];
    }
  }
  Status CreateResource(SqlIterableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new SqlIterableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
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

    const Tensor* field_tensor;
    OP_REQUIRES_OK(context, context->input("field", &field_tensor));

    OP_REQUIRES_OK(
        context,
        resource->Read(index, field_tensor,
                       [&](const int64 field_index, Tensor** value) -> Status {
                         TF_RETURN_IF_ERROR(context->allocate_output(
                             field_index, TensorShape({1}), value));
                         return Status::OK();
                       }));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>SqlIterableInit").Device(DEVICE_CPU),
                        SqlIterableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>SqlIterableRead").Device(DEVICE_CPU),
                        SqlIterableReadOp);
}  // namespace
}  // namespace io
}  // namespace tensorflow
