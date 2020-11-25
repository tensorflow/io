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

class MongoReadableResource : public ResourceBase {
 public:
  MongoReadableResource(Env* env) : env_(env) {}
  ~MongoReadableResource() {
    mongoc_collection_destroy(collection_);
    mongoc_database_destroy(database_);
    mongoc_uri_destroy(uri_);
    mongoc_client_destroy(client_);
    mongoc_cleanup();
  }

  Status Init() {
    //   Required to initialize libmongoc's internals
    mongoc_init();

    // Log the uri
    LOG(ERROR) << uri_string;

    // Create a MongoDB URI object from the given string

    uri_ = mongoc_uri_new_with_error(uri_string, &error_);
    if (!uri_) {
      return errors::FailedPrecondition("Failed to parse URI: ", uri_string,
                                        "due to: ", error_.message);
    }

    // Initialize the MongoDB client

    client_ = mongoc_client_new_from_uri(uri_);
    if (!client_) {
      return errors::FailedPrecondition("Failed to initialize the client");
    }

    // Register the application name so we can track it in the profile logs
    //  on the server. This can also be done from the URI.

    mongoc_client_set_appname(client_, "tfio-mongo");

    //  Get a handle on the database "db_name" and collection "coll_name"

    database_ = mongoc_client_get_database(client_, "tfiodb");
    collection_ = mongoc_client_get_collection(client_, "tfiodb", "test");

    // Perform healthcheck before proceeding
    Healthcheck();
  }

  string DebugString() const override { return "MongoReadableResource"; }

 protected:
  Status Healthcheck() {
    // Ping the server to ensure connectivity

    cmd_ = BCON_NEW("ping", BCON_INT32(1));

    retval_ = mongoc_client_command_simple(client_, "admin", cmd_, NULL,
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
  const char* uri_string = "mongodb://mongoadmin:secret@localhost:27017";
  mongoc_uri_t* uri_;
  mongoc_client_t* client_;
  mongoc_database_t* database_;
  mongoc_collection_t* collection_;
  bson_t *cmd_, reply_;
  bson_error_t error_;
  char* str;
  bool retval_;
};

class MongoReadableInitOp : public ResourceOpKernel<MongoReadableResource> {
 public:
  explicit MongoReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<MongoReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<MongoReadableResource>::Compute(context);

    OP_REQUIRES_OK(context, resource_->Init());
  }

  Status CreateResource(MongoReadableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new MongoReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>MongoReadableInit").Device(DEVICE_CPU),
                        MongoReadableInitOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
