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
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
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

  Status Init(const std::string& healthcheck_url,
              const std::string& healthcheck_field,
              const std::string& request_url,
              std::function<Status(const TensorShape& columns_shape,
                                   Tensor** columns, Tensor** dtypes)>
                  allocate_func) {
    // Perform healthcheck before proceeding
    Healthcheck(healthcheck_url, healthcheck_field);

    // Make the request API call and set the metadata based on a sample of
    // data returned. The request_url will have the "scroll" param set with
    // a very small value (approx. 1ms) so that the response is immediate
    // and the metadata can be retrieved quickly.
    base_dtypes_.clear();
    base_columns_.clear();
    rapidjson::Document response_json;
    MakeAPICall(request_url, &response_json);

    // Validate the presence of the _scroll_id in the response.
    // The _scroll_id keeps might change in subsequent calls, thus not
    // setting it as a resource attribute.

    if (!response_json.HasMember("_scroll_id")) {
      return errors::FailedPrecondition(
          "Failed to start the scrolling search context.");
    }

    if (response_json.HasMember("hits")) {
      const rapidjson::Value& hits = response_json["hits"]["hits"].GetArray();
      // Throw an error if empty list is returned by the cluster.
      if (hits.MemberCount() < 1) {
        return errors::OutOfRange("Empty hits returned by cluster");
      }

      // Capture and validate dtype and column/field information by
      // evaluating the first item in the returned hits.
      const rapidjson::Value& last_item = hits[0]["_source"];

      for (rapidjson::Value::ConstMemberIterator itr = last_item.MemberBegin();
           itr != last_item.MemberEnd(); ++itr) {
        DataType dtype;
        if (itr->value.IsInt64()) {
          dtype = DT_INT64;
        } else if (itr->value.IsInt()) {
          dtype = DT_INT32;
        } else if (itr->value.IsDouble()) {
          dtype = DT_DOUBLE;
        } else if (itr->value.IsString()) {
          dtype = DT_STRING;
        } else {
          return errors::InvalidArgument(
              "field: ", itr->name.GetString(),
              "has unsupported data type: ", itr->value.GetType());
        }

        base_dtypes_.push_back(dtype);
        base_columns_.push_back(itr->name.GetString());
      }

      TensorShape columns_shape({static_cast<int64>(base_columns_.size())});
      Tensor* columns_tensor;
      Tensor* dtypes_tensor;
      TF_RETURN_IF_ERROR(
          allocate_func(columns_shape, &columns_tensor, &dtypes_tensor));
      for (int column_idx = 0; column_idx < base_columns_.size();
           ++column_idx) {
        columns_tensor->flat<tstring>()(column_idx) = base_columns_[column_idx];
        if (base_dtypes_[column_idx] == DT_INT64) {
          dtypes_tensor->flat<tstring>()(column_idx) = "DT_INT64";
        } else if (base_dtypes_[column_idx] == DT_INT32) {
          dtypes_tensor->flat<tstring>()(column_idx) = "DT_INT32";
        } else if (base_dtypes_[column_idx] == DT_DOUBLE) {
          dtypes_tensor->flat<tstring>()(column_idx) = "DT_DOUBLE";
        } else if (base_dtypes_[column_idx] == DT_STRING) {
          dtypes_tensor->flat<tstring>()(column_idx) = "DT_STRING";
        }
      }

    } else {
      return errors::FailedPrecondition("Corrupted response from the server");
    }

    return Status::OK();
  }

  Status Next(
      const std::string& request_url, const std::string& scroll_request_url,
      std::function<Status(const TensorShape& tensor_shape, Tensor** items)>
          data_allocate_func) {
    rapidjson::Document response_json;
    if (scroll_id == "") {
      MakeAPICall(request_url, &response_json);
    } else {
      MakeAPICall(scroll_request_url, &response_json);
    }

    if (response_json.HasMember("_scroll_id")) {
      scroll_id = response_json["_scroll_id"].GetString();
    } else {
      scroll_id = "";
    }

    if (response_json.HasMember("hits")) {
      const rapidjson::Value& hits = response_json["hits"]["hits"].GetArray();
      // LOG(INFO) << "Response has hits= " << hits.MemberCount();
      if (hits.MemberCount() > 0) {
        TensorShape tensor_shape({static_cast<int64>(hits.MemberCount())});
        Tensor* items;
        TF_RETURN_IF_ERROR(data_allocate_func(tensor_shape, &items));

        for (size_t item_idx = 0; item_idx < hits.MemberCount(); ++item_idx) {
          const rapidjson::Value& value = hits[item_idx]["_source"];
          rapidjson::StringBuffer item_buffer;
          item_buffer.Clear();
          rapidjson::Writer<rapidjson::StringBuffer> item_writer(item_buffer);
          value.Accept(item_writer);
          items->flat<tstring>()(item_idx) = item_buffer.GetString();
        }

      } else {
        scroll_id = "";
        TensorShape tensor_shape({static_cast<int64>(hits.MemberCount())});
        Tensor* items;
        TF_RETURN_IF_ERROR(data_allocate_func(tensor_shape, &items));
      }
    } else {
      rapidjson::StringBuffer error_buffer;
      error_buffer.Clear();
      rapidjson::Writer<rapidjson::StringBuffer> error_writer(error_buffer);
      response_json.Accept(error_writer);
      std::string error_response = error_buffer.GetString();
      return errors::FailedPrecondition("Corrupted response from the server " +
                                        error_response);
    }

    return Status::OK();
  }

  string DebugString() const override { return "ElasticsearchBaseResource"; }

 protected:
  Status Healthcheck(const std::string& healthcheck_url,
                     const std::string& healthcheck_field) {
    // Make the healthcheck API call and get the response json
    rapidjson::Document response_json;
    MakeAPICall(healthcheck_url, &response_json);

    if (response_json.HasMember(healthcheck_field.c_str())) {
      // LOG(INFO) << "cluster health: "
      //           << response_json[healthcheck_field.c_str()].GetString();
    } else
      return errors::FailedPrecondition("healthcheck failed");

    return Status::OK();
  }

  Status MakeAPICall(const std::string& url,
                     rapidjson::Document* response_json) {
    HttpRequest* request = http_request_factory_.Create();

    if (scroll_id != "") {
      std::string scroll_url = url + "?scroll=1ms&scroll_id=" + scroll_id;
      // LOG(INFO) << "Setting the url" << scroll_url;
      request->SetUri(scroll_url);
    } else {
      // LOG(INFO) << "Setting the url" << url;
      request->SetUri(url);
    }

    // LOG(INFO) << "Setting the headers";
    request->AddHeader("Content-Type", "application/json; charset=utf-8");

    // LOG(INFO) << "Setting the response buffer";
    std::vector<char> response;
    request->SetResultBuffer(&response);

    // LOG(INFO) << "Sending the request";
    TF_RETURN_IF_ERROR(request->Send());

    // LOG(INFO) << "Response code" << request->GetResponseCode();

    // LOG(INFO) << "Getting the length of content";
    string length_string = request->GetResponseHeader("content-length");
    // LOG(INFO) << "response length: " << length_string;

    std::string response_str(response.begin(), response.end());
    // LOG(INFO) << "response " << response_str;

    if (response_json->Parse(response_str.c_str()).HasParseError()) {
      LOG(ERROR) << "Error while parsing json at offset: "
                 << response_json->GetErrorOffset() << " : "
                 << GetParseError_En(response_json->GetParseError());
      return errors::InvalidArgument(
          "Unable to convert the response body to JSON");
    }

    if (!response_json->IsObject()) {
      return errors::InvalidArgument(
          "Invalid JSON response. The response should be an object");
    }

    return Status::OK();
  }

  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  string url_;
  CurlHttpRequest::Factory http_request_factory_ = CurlHttpRequest::Factory();

  std::vector<DataType> base_dtypes_;
  std::vector<string> base_columns_;
  std::string scroll_id = "";
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

    const Tensor* healthcheck_url_tensor;
    OP_REQUIRES_OK(context,
                   context->input("healthcheck_url", &healthcheck_url_tensor));
    const string& healthcheck_url = healthcheck_url_tensor->scalar<tstring>()();

    const Tensor* healthcheck_field_tensor;
    OP_REQUIRES_OK(context, context->input("healthcheck_field",
                                           &healthcheck_field_tensor));
    const string& healthcheck_field =
        healthcheck_field_tensor->scalar<tstring>()();

    const Tensor* request_url_tensor;
    OP_REQUIRES_OK(context, context->input("request_url", &request_url_tensor));
    const string& request_url = request_url_tensor->scalar<tstring>()();

    OP_REQUIRES_OK(
        context,
        resource_->Init(healthcheck_url, healthcheck_field, request_url,
                        [&](const TensorShape& columns_shape, Tensor** columns,
                            Tensor** dtypes) -> Status {
                          TF_RETURN_IF_ERROR(context->allocate_output(
                              1, columns_shape, columns));
                          TF_RETURN_IF_ERROR(context->allocate_output(
                              2, columns_shape, dtypes));
                          return Status::OK();
                        }));
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

class ElasticsearchReadableNextOp : public OpKernel {
 public:
  explicit ElasticsearchReadableNextOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    ElasticsearchReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* request_url_tensor;
    OP_REQUIRES_OK(context, context->input("request_url", &request_url_tensor));
    const string& request_url = request_url_tensor->scalar<tstring>()();

    const Tensor* scroll_request_url_tensor;
    OP_REQUIRES_OK(context, context->input("scroll_request_url",
                                           &scroll_request_url_tensor));
    const string& scroll_request_url =
        scroll_request_url_tensor->scalar<tstring>()();

    OP_REQUIRES_OK(context,
                   resource->Next(request_url, scroll_request_url,
                                  [&](const TensorShape& tensor_shape,
                                      Tensor** items) -> Status {
                                    TF_RETURN_IF_ERROR(context->allocate_output(
                                        0, tensor_shape, items));
                                    return Status::OK();
                                  }));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IO>ElasticsearchReadableInit").Device(DEVICE_CPU),
                        ElasticsearchReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>ElasticsearchReadableNext").Device(DEVICE_CPU),
                        ElasticsearchReadableNextOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow
