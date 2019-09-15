/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow_io/kinesis/kernels/aws_kernels.h"
#include <aws/s3/S3Client.h>
#include <aws/s3/S3Errors.h>
#include <aws/s3/model/ListObjectsRequest.h>

namespace tensorflow {
namespace data {

static const char* kS3AllocationTag = "S3Allocation";
static const int kS3MaxKeys = 100;

Status ParseS3Path(const string& fname, bool empty_object_ok, string* bucket,
                   string* object) {
  if (!bucket || !object) {
    return errors::Internal("bucket and object cannot be null.");
  }
  StringPiece scheme, bucketp, objectp;
  io::ParseURI(fname, &scheme, &bucketp, &objectp);
  if (scheme != "s3") {
    return errors::InvalidArgument("S3 path doesn't start with 's3://': ",
                                   fname);
  }
  *bucket = string(bucketp);
  if (bucket->empty() || *bucket == ".") {
    return errors::InvalidArgument("S3 path doesn't contain a bucket name: ",
                                   fname);
  }
  absl::ConsumePrefix(&objectp, "/");
  *object = string(objectp);
  if (!empty_object_ok && object->empty()) {
    return errors::InvalidArgument("S3 path doesn't contain an object name: ",
                                   fname);
  }
  return Status::OK();
}

Aws::Client::ClientConfiguration* InitializeDefaultS3ClientConfig() {
  static Aws::Client::ClientConfiguration config;
  const char* endpoint = getenv("S3_ENDPOINT");
  if (endpoint) {
    config.endpointOverride = Aws::String(endpoint);
  }
  const char* region = getenv("AWS_REGION");
  if (region) {
    config.region = Aws::String(region);
  } else {
    // Load config file (e.g., ~/.aws/config) only if AWS_SDK_LOAD_CONFIG
    // is set with a truthy value.
    const char* load_config_env = getenv("AWS_SDK_LOAD_CONFIG");
    string load_config =
        load_config_env ? str_util::Lowercase(load_config_env) : "";
    if (load_config == "true" || load_config == "1") {
      Aws::String config_file;
      // If AWS_CONFIG_FILE is set then use it, otherwise use ~/.aws/config.
      const char* config_file_env = getenv("AWS_CONFIG_FILE");
      if (config_file_env) {
        config_file = config_file_env;
      } else {
        const char* home_env = getenv("HOME");
        if (home_env) {
          config_file = home_env;
          config_file += "/.aws/config";
        }
      }
      Aws::Config::AWSConfigFileProfileConfigLoader loader(config_file);
      // Load the configuration. If successful, get the region.
      // If the load is not successful, then generate a warning.
      if (loader.Load()) {
        auto profiles = loader.GetProfiles();
        if (!profiles["default"].GetRegion().empty()) {
          config.region = profiles["default"].GetRegion();
        }
      } else {
        LOG(WARNING) << "Failed to load the profile in " << config_file << ".";
      }
    }
  }
  const char* use_https = getenv("S3_USE_HTTPS");
  if (use_https) {
    if (use_https[0] == '0') {
      config.scheme = Aws::Http::Scheme::HTTP;
    } else {
      config.scheme = Aws::Http::Scheme::HTTPS;
    }
  }
  const char* verify_ssl = getenv("S3_VERIFY_SSL");
  if (verify_ssl) {
    if (verify_ssl[0] == '0') {
      config.verifySSL = false;
    } else {
      config.verifySSL = true;
    }
  }
  const char* connect_timeout = getenv("S3_CONNECT_TIMEOUT_MSEC");
  if (connect_timeout) {
    int64 timeout;

    if (strings::safe_strto64(connect_timeout, &timeout)) {
      config.connectTimeoutMs = timeout;
    }
  }
  const char* request_timeout = getenv("S3_REQUEST_TIMEOUT_MSEC");
  if (request_timeout) {
    int64 timeout;

    if (strings::safe_strto64(request_timeout, &timeout)) {
      config.requestTimeoutMs = timeout;
    }
  }

  return &config;
}

Aws::Client::ClientConfiguration& GetDefaultS3ClientConfig() {
  static Aws::Client::ClientConfiguration* config =
      InitializeDefaultS3ClientConfig();
  return *config;
}
class StorageListS3Op : public OpKernel {
 public:
  explicit StorageListS3Op(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const string& input = input_tensor.scalar<string>()();
    string bucket, prefix;
    OP_REQUIRES_OK(context, ParseS3Path(input, false, &bucket, &prefix));

    if (prefix.back() != '/') {
      prefix.push_back('/');
    }

    AwsInitAPI();

    // The creation of S3Client disables virtual addressing:
    //   S3Client(clientConfiguration, signPayloads, useVirtualAdressing = true)
    // The purpose is to address the issue encountered when there is an `.`
    // in the bucket name. Due to TLS hostname validation or DNS rules,
    // the bucket may not be resolved. Disabling of virtual addressing
    // should address the issue. See GitHub issue 16397 for details.
    std::shared_ptr<Aws::S3::S3Client> client = std::shared_ptr<Aws::S3::S3Client>(new Aws::S3::S3Client(
        GetDefaultS3ClientConfig(),
        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false));

    Aws::S3::Model::ListObjectsRequest listObjectsRequest;
    listObjectsRequest.WithBucket(bucket.c_str())
        .WithPrefix(prefix.c_str())
        .WithMaxKeys(kS3MaxKeys);
    listObjectsRequest.SetResponseStreamFactory(
        []() { return Aws::New<Aws::StringStream>(kS3AllocationTag); });

    std::vector<string> result;
    Aws::S3::Model::ListObjectsResult listObjectsResult;
    do {
      auto listObjectsOutcome =
        client->ListObjects(listObjectsRequest);
      OP_REQUIRES(context, listObjectsOutcome.IsSuccess(),
          errors::Unknown(listObjectsOutcome.GetError().GetExceptionName(), ": ", listObjectsOutcome.GetError().GetMessage()));
    
      listObjectsResult = listObjectsOutcome.GetResult();
      for (const auto& object : listObjectsResult.GetContents()) {
        Aws::String s = object.GetKey();
        Aws::String entry = s.substr(strlen(prefix.c_str()));
        if (entry.length() > 0) {
          result.push_back(entry.c_str());
        }
      }
      listObjectsRequest.SetMarker(listObjectsResult.GetNextMarker());
    } while (listObjectsResult.GetIsTruncated());

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({static_cast<int64>(result.size())}), &output_tensor));
    for (size_t i = 0; i < result.size(); i++) {
      string value = "s3://" + bucket + "/" + prefix + result[i];
      output_tensor->flat<string>()(i) = value;
    }
  }
 private:
  mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
REGISTER_KERNEL_BUILDER(Name("StorageListS3").Device(DEVICE_CPU),
                        StorageListS3Op);
}  // namespace data
}  // namespace tensorflow
