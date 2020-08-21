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

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/cloud/curl_http_request.h"

namespace tensorflow {
namespace io {
namespace {

class ElasticsearchReadableResource : public ResourceBase {
 public:
  ElasticsearchReadableResource(Env* env) : env_(env) {}
  ~ElasticsearchReadableResource() {}

  Status Init(const std::string& url, const std::string& healthcheck_field) {
    HttpRequest* request = http_request_factory_.Create();
    // LOG(INFO) << "Setting the url";
    request->SetUri(url);

    // LOG(INFO) << "Setting the response buffer";
    std::vector<char> response;
    request->SetResultBuffer(&response);

    // LOG(INFO) << "Sending the request";
    TF_RETURN_IF_ERROR(request->Send());

    // LOG(INFO) << "Getting the length of content";
    string length_string = request->GetResponseHeader("content-length");

    // LOG(INFO) << "response length: " << length_string;
    std::string response_str(response.begin(), response.end());

    // LOG(INFO) << "response: " << response_str;
    rapidjson::Document response_json;

    if (response_json.Parse(response_str.c_str()).HasParseError()) {
      LOG(ERROR) << "Error while parsing json at offset: "
                 << response_json.GetErrorOffset() << " : "
                 << GetParseError_En(response_json.GetParseError());
      return errors::InvalidArgument(
          "Unable to convert the response body to JSON");
    }

    if (response_json.HasMember(healthcheck_field.c_str())) {
      LOG(INFO) << "cluster health: "
                << response_json[healthcheck_field.c_str()].GetString();
    } else
      return errors::FailedPrecondition("healthcheck failed");

    return Status::OK();
  }

  Status Read() {
    // TODO (kvignesh1420)
  }

  string DebugString() const override { return "ElasticsearchBaseResource"; }

 protected:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  string url_;
  CurlHttpRequest::Factory http_request_factory_ = CurlHttpRequest::Factory();
};

class ElasticsearchReadableInitOp
    : public ResourceOpKernel<ElasticsearchReadableResource> {
 public:
  explicit ElasticsearchReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<ElasticsearchReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<ElasticsearchReadableResource>::Compute(context);

    const Tensor* url_tensor;
    OP_REQUIRES_OK(context, context->input("url", &url_tensor));
    const string& url = url_tensor->scalar<tstring>()();

    const Tensor* healthcheck_field_tensor;
    OP_REQUIRES_OK(context, context->input("healthcheck_field",
                                           &healthcheck_field_tensor));
    const string& healthcheck_field =
        healthcheck_field_tensor->scalar<tstring>()();

    OP_REQUIRES_OK(context, resource_->Init(url, healthcheck_field));
  }

  Status CreateResource(ElasticsearchReadableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new ElasticsearchReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>ElasticsearchReadableInit").Device(DEVICE_CPU),
                        ElasticsearchReadableInitOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
