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
    LOG(ERROR) << uri;

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

    // Register the application name so we can track it in the profile logs
    //  on the server. This can also be done from the URI.

    mongoc_client_set_appname(client_obj_, "tfio-mongo");

    //  Get a handle on the database "db_name" and collection "coll_name"

    database_obj_ = mongoc_client_get_database(client_obj_, database.c_str());
    collection_obj_ = mongoc_client_get_collection(
        client_obj_, database.c_str(), collection.c_str());

    // Perform healthcheck before proceeding
    Healthcheck();
  }

  string DebugString() const override { return "MongoDBReadableResource"; }

 protected:
  Status Healthcheck() {
    // Ping the server to ensure connectivity

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

REGISTER_KERNEL_BUILDER(Name("IO>MongoDBReadableInit").Device(DEVICE_CPU),
                        MongoDBReadableInitOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
