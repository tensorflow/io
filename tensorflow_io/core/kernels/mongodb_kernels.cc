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

#include <bson/bson.h>
#include <mongoc/mongoc.h>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {
namespace io {
namespace {

class MongoDBReadableResource : public ResourceBase {
 public:
  MongoDBReadableResource(Env* env) : env_(env) {}
  ~MongoDBReadableResource() {
    mongoc_collection_destroy(collection_obj_);
    mongoc_database_destroy(database_obj_);
    mongoc_uri_destroy(uri_obj_);
    mongoc_client_destroy(client_obj_);
    mongoc_cleanup();
  }

  Status Init(const std::string& uri, const std::string& database,
              const std::string& collection) {
    //   Required to initialize libmongoc's internals
    mongoc_init();

    // Log the uri
    LOG(INFO) << "Connecting to: " << uri;

    // Create a MongoDB URI object from the given string

    uri_obj_ = mongoc_uri_new_with_error(uri.c_str(), &error_);
    if (!uri_obj_) {
      return errors::FailedPrecondition("Failed to parse URI: ", uri,
                                        "due to: ", error_.message);
    }

    // Initialize the MongoDB client

    client_obj_ = mongoc_client_new_from_uri(uri_obj_);
    if (!client_obj_) {
      return errors::FailedPrecondition("Failed to initialize the client");
    }

    //  Get a handle on the database "db_name" and collection "coll_name"

    database_obj_ = mongoc_client_get_database(client_obj_, database.c_str());
    collection_obj_ = mongoc_client_get_collection(
        client_obj_, database.c_str(), collection.c_str());

    cursor_obj_ =
        mongoc_collection_find_with_opts(collection_obj_, query_, NULL, NULL);

    // Perform healthcheck before proceeding
    Healthcheck();
    return Status::OK();
  }

  Status Next(std::function<Status(const TensorShape& shape, Tensor** record)>
                  allocate_func) {
    mutex_lock l(mu_);

    constexpr size_t max_num_records = 1024;
    std::vector<std::string> records;
    records.reserve(max_num_records);

    const bson_t* doc;

    int num_records = 0;
    while (num_records < max_num_records) {
      if (mongoc_cursor_next(cursor_obj_, &doc)) {
        // Reference for BSON to JSON conversion:
        // https://github.com/mongodb/specifications/blob/master/source/extended-json.rst#conversion-table
        char* record = bson_as_relaxed_extended_json(doc, NULL);
        records.emplace_back(record);
        num_records++;
        bson_free(record);
      } else {
        break;
      }
    }

    if (records.size() == 0) {
      // resetting the cursor after reaching the end of the collection.
      cursor_obj_ =
          mongoc_collection_find_with_opts(collection_obj_, query_, NULL, NULL);
    }
    TensorShape shape({static_cast<int32>(records.size())});
    Tensor* records_tensor;
    TF_RETURN_IF_ERROR(allocate_func(shape, &records_tensor));

    // If no messages were received when timeout exceeded, we treat it as a
    // failure and don't continue receiving messages.
    for (size_t i = 0; i < records.size(); i++) {
      records_tensor->flat<tstring>()(i) = records[i];
    }

    return Status::OK();
  }

  string DebugString() const override { return "MongoDBReadableResource"; }

 protected:
  Status Healthcheck() {
    // Ping the server to check connectivity

    cmd_ = BCON_NEW("ping", BCON_INT32(1));

    retval_ = mongoc_client_command_simple(client_obj_, "admin", cmd_, NULL,
                                           &reply_, &error_);

    if (!retval_) {
      return errors::FailedPrecondition(
          "Failed to ping the mongo cluster due to: ", error_.message);
    }
    LOG(ERROR) << "Ping Successful";

    return Status::OK();
  }

  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  mongoc_uri_t* uri_obj_;
  mongoc_client_t* client_obj_;
  mongoc_database_t* database_obj_;
  mongoc_collection_t* collection_obj_;
  mongoc_cursor_t* cursor_obj_;
  bson_t* query_ = bson_new();
  bson_t *cmd_, reply_;
  bson_error_t error_;
  char* str;
  bool retval_;
};

class MongoDBReadableInitOp : public ResourceOpKernel<MongoDBReadableResource> {
 public:
  explicit MongoDBReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<MongoDBReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<MongoDBReadableResource>::Compute(context);

    const Tensor* uri_tensor;
    OP_REQUIRES_OK(context, context->input("uri", &uri_tensor));
    const string& uri = uri_tensor->scalar<tstring>()();

    const Tensor* database_tensor;
    OP_REQUIRES_OK(context, context->input("database", &database_tensor));
    const string& database = database_tensor->scalar<tstring>()();

    const Tensor* collection_tensor;
    OP_REQUIRES_OK(context, context->input("collection", &collection_tensor));
    const string& collection = collection_tensor->scalar<tstring>()();

    OP_REQUIRES_OK(context, resource_->Init(uri, database, collection));
  }

  Status CreateResource(MongoDBReadableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new MongoDBReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class MongoDBReadableNextOp : public OpKernel {
 public:
  explicit MongoDBReadableNextOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    MongoDBReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);

    OP_REQUIRES_OK(context, resource->Next([&](const TensorShape& shape,
                                               Tensor** record) -> Status {
      TF_RETURN_IF_ERROR(context->allocate_output(0, shape, record));
      return Status::OK();
    }));
  }

 private:
  mutable mutex mu_;
};

class MongoDBWritableResource : public ResourceBase {
 public:
  MongoDBWritableResource(Env* env) : env_(env) {}
  ~MongoDBWritableResource() {
    mongoc_collection_destroy(collection_obj_);
    mongoc_database_destroy(database_obj_);
    mongoc_uri_destroy(uri_obj_);
    mongoc_client_destroy(client_obj_);
    mongoc_cleanup();
  }

  Status Init(const std::string& uri, const std::string& database,
              const std::string& collection) {
    //   Required to initialize libmongoc's internals
    mongoc_init();

    // Log the uri
    LOG(INFO) << "Connecting to: " << uri;

    // Create a MongoDB URI object from the given string

    uri_obj_ = mongoc_uri_new_with_error(uri.c_str(), &error_);
    if (!uri_obj_) {
      return errors::FailedPrecondition("Failed to parse URI: ", uri,
                                        "due to: ", error_.message);
    }

    // Initialize the MongoDB client

    client_obj_ = mongoc_client_new_from_uri(uri_obj_);
    if (!client_obj_) {
      return errors::FailedPrecondition("Failed to initialize the client");
    }

    //  Get a handle on the database "db_name" and collection "coll_name"

    database_obj_ = mongoc_client_get_database(client_obj_, database.c_str());
    collection_obj_ = mongoc_client_get_collection(
        client_obj_, database.c_str(), collection.c_str());

    // Perform healthcheck before proceeding
    Healthcheck();
    return Status::OK();
  }

  Status Write(const std::string& record) {
    const char* json_record = record.c_str();
    bson_t* bson_record =
        bson_new_from_json((const uint8_t*)json_record, -1, &error_);

    if (!bson_record) {
      return errors::FailedPrecondition("Failed to parse json due to: ",
                                        error_.message);
    }

    if (!mongoc_collection_insert_one(collection_obj_, bson_record, NULL, NULL,
                                      &error_)) {
      return errors::FailedPrecondition("Failed to insert document due to: ",
                                        error_.message);
    }

    bson_destroy(bson_record);

    return Status::OK();
  }

  Status DeleteMany(const std::string& record) {
    const char* json_record = record.c_str();
    bson_t* bson_record =
        bson_new_from_json((const uint8_t*)json_record, -1, &error_);

    if (!bson_record) {
      return errors::FailedPrecondition("Failed to parse json due to: ",
                                        error_.message);
    }

    if (!mongoc_collection_delete_many(collection_obj_, bson_record, NULL, NULL,
                                       &error_)) {
      return errors::FailedPrecondition(
          "Failed to delete matching documents due to: ", error_.message);
    }

    bson_destroy(bson_record);

    return Status::OK();
  }

  string DebugString() const override { return "MongoDBWritableResource"; }

 protected:
  Status Healthcheck() {
    // Ping the server to check connectivity

    cmd_ = BCON_NEW("ping", BCON_INT32(1));

    retval_ = mongoc_client_command_simple(client_obj_, "admin", cmd_, NULL,
                                           &reply_, &error_);

    if (!retval_) {
      return errors::FailedPrecondition(
          "Failed to ping the mongo cluster due to: ", error_.message);
    }
    LOG(INFO) << "Ping Successful";

    return Status::OK();
  }

  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  mongoc_uri_t* uri_obj_;
  mongoc_client_t* client_obj_;
  mongoc_database_t* database_obj_;
  mongoc_collection_t* collection_obj_;
  bson_t *cmd_, reply_;
  bson_error_t error_;
  char* str;
  bool retval_;
};

class MongoDBWritableInitOp : public ResourceOpKernel<MongoDBWritableResource> {
 public:
  explicit MongoDBWritableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<MongoDBWritableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<MongoDBWritableResource>::Compute(context);

    const Tensor* uri_tensor;
    OP_REQUIRES_OK(context, context->input("uri", &uri_tensor));
    const string& uri = uri_tensor->scalar<tstring>()();

    const Tensor* database_tensor;
    OP_REQUIRES_OK(context, context->input("database", &database_tensor));
    const string& database = database_tensor->scalar<tstring>()();

    const Tensor* collection_tensor;
    OP_REQUIRES_OK(context, context->input("collection", &collection_tensor));
    const string& collection = collection_tensor->scalar<tstring>()();

    OP_REQUIRES_OK(context, resource_->Init(uri, database, collection));
  }

  Status CreateResource(MongoDBWritableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new MongoDBWritableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class MongoDBWritableWriteOp : public OpKernel {
 public:
  explicit MongoDBWritableWriteOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    MongoDBWritableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* record_tensor;
    OP_REQUIRES_OK(context, context->input("record", &record_tensor));
    const string& record = record_tensor->scalar<tstring>()();

    OP_REQUIRES_OK(context, resource->Write(record));
  }

 private:
  mutable mutex mu_;
};

class MongoDBWritableDeleteManyOp : public OpKernel {
 public:
  explicit MongoDBWritableDeleteManyOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    MongoDBWritableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* record_tensor;
    OP_REQUIRES_OK(context, context->input("record", &record_tensor));
    const string& record = record_tensor->scalar<tstring>()();

    OP_REQUIRES_OK(context, resource->DeleteMany(record));
  }

 private:
  mutable mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("IO>MongoDBReadableInit").Device(DEVICE_CPU),
                        MongoDBReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>MongoDBReadableNext").Device(DEVICE_CPU),
                        MongoDBReadableNextOp);
REGISTER_KERNEL_BUILDER(Name("IO>MongoDBWritableInit").Device(DEVICE_CPU),
                        MongoDBWritableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>MongoDBWritableWrite").Device(DEVICE_CPU),
                        MongoDBWritableWriteOp);
REGISTER_KERNEL_BUILDER(Name("IO>MongoDBWritableDeleteMany").Device(DEVICE_CPU),
                        MongoDBWritableDeleteManyOp);
}  // namespace
}  // namespace io
}  // namespace tensorflow
