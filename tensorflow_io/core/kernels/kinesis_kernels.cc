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

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

#include <openssl/hmac.h>
#include <openssl/sha.h>

#include <aws/core/Aws.h>
#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/core/utils/Outcome.h>
#include <aws/core/utils/crypto/Factories.h>
#include <aws/core/utils/crypto/HMAC.h>
#include <aws/core/utils/crypto/Hash.h>
#include <aws/core/utils/crypto/HashResult.h>
#include <aws/kinesis/KinesisClient.h>
#include <aws/kinesis/model/DescribeStreamRequest.h>
#include <aws/kinesis/model/GetRecordsRequest.h>
#include <aws/kinesis/model/GetShardIteratorRequest.h>
#include <aws/kinesis/model/PutRecordsRequest.h>
#include <aws/kinesis/model/ShardIteratorType.h>

namespace tensorflow {
namespace data {
namespace {

static const char* AWSCryptoAllocationTag = "AWSCryptoAllocation";

class AWSSHA256Factory : public Aws::Utils::Crypto::HashFactory {
 public:
  std::shared_ptr<Aws::Utils::Crypto::Hash> CreateImplementation()
      const override;
};

class AWSSHA256HmacFactory : public Aws::Utils::Crypto::HMACFactory {
 public:
  std::shared_ptr<Aws::Utils::Crypto::HMAC> CreateImplementation()
      const override;
};

class AWSSha256HMACOpenSSLImpl : public Aws::Utils::Crypto::HMAC {
 public:
  AWSSha256HMACOpenSSLImpl() {}

  virtual ~AWSSha256HMACOpenSSLImpl() = default;

  virtual Aws::Utils::Crypto::HashResult Calculate(
      const Aws::Utils::ByteBuffer& toSign,
      const Aws::Utils::ByteBuffer& secret) override {
    unsigned int length = SHA256_DIGEST_LENGTH;
    Aws::Utils::ByteBuffer digest(length);
    memset(digest.GetUnderlyingData(), 0, length);

    HMAC_CTX ctx;
    HMAC_CTX_init(&ctx);

    HMAC_Init_ex(&ctx, secret.GetUnderlyingData(),
                 static_cast<int>(secret.GetLength()), EVP_sha256(), NULL);
    HMAC_Update(&ctx, toSign.GetUnderlyingData(), toSign.GetLength());
    HMAC_Final(&ctx, digest.GetUnderlyingData(), &length);
    HMAC_CTX_cleanup(&ctx);

    return Aws::Utils::Crypto::HashResult(std::move(digest));
  }
};

class AWSSha256OpenSSLImpl : public Aws::Utils::Crypto::Hash {
 public:
  AWSSha256OpenSSLImpl() {}

  virtual ~AWSSha256OpenSSLImpl() = default;

  virtual Aws::Utils::Crypto::HashResult Calculate(
      const Aws::String& str) override {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str.data(), str.size());

    Aws::Utils::ByteBuffer hash(SHA256_DIGEST_LENGTH);
    SHA256_Final(hash.GetUnderlyingData(), &sha256);

    return Aws::Utils::Crypto::HashResult(std::move(hash));
  }

  virtual Aws::Utils::Crypto::HashResult Calculate(
      Aws::IStream& stream) override {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);

    auto currentPos = stream.tellg();
    if (currentPos == std::streampos(std::streamoff(-1))) {
      currentPos = 0;
      stream.clear();
    }

    stream.seekg(0, stream.beg);

    char streamBuffer
        [Aws::Utils::Crypto::Hash::INTERNAL_HASH_STREAM_BUFFER_SIZE];
    while (stream.good()) {
      stream.read(streamBuffer,
                  Aws::Utils::Crypto::Hash::INTERNAL_HASH_STREAM_BUFFER_SIZE);
      auto bytesRead = stream.gcount();

      if (bytesRead > 0) {
        SHA256_Update(&sha256, streamBuffer, static_cast<size_t>(bytesRead));
      }
    }

    stream.clear();
    stream.seekg(currentPos, stream.beg);

    Aws::Utils::ByteBuffer hash(SHA256_DIGEST_LENGTH);
    SHA256_Final(hash.GetUnderlyingData(), &sha256);

    return Aws::Utils::Crypto::HashResult(std::move(hash));
  }
};

std::shared_ptr<Aws::Utils::Crypto::Hash>
AWSSHA256Factory::CreateImplementation() const {
  return Aws::MakeShared<AWSSha256OpenSSLImpl>(AWSCryptoAllocationTag);
}

std::shared_ptr<Aws::Utils::Crypto::HMAC>
AWSSHA256HmacFactory::CreateImplementation() const {
  return Aws::MakeShared<AWSSha256HMACOpenSSLImpl>(AWSCryptoAllocationTag);
}

Aws::Client::ClientConfiguration* InitializeDefaultClientConfig() {
  static Aws::Client::ClientConfiguration config;
  const char* endpoint = getenv("KINESIS_ENDPOINT");
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
  const char* use_https = getenv("KINESIS_USE_HTTPS");
  if (use_https) {
    if (use_https[0] == '0') {
      config.scheme = Aws::Http::Scheme::HTTP;
    } else {
      config.scheme = Aws::Http::Scheme::HTTPS;
    }
  }
  const char* verify_ssl = getenv("KINESIS_VERIFY_SSL");
  if (verify_ssl) {
    if (verify_ssl[0] == '0') {
      config.verifySSL = false;
    } else {
      config.verifySSL = true;
    }
  }
  const char* connect_timeout = getenv("KINESIS_CONNECT_TIMEOUT_MSEC");
  if (connect_timeout) {
    int64 timeout;

    if (strings::safe_strto64(connect_timeout, &timeout)) {
      config.connectTimeoutMs = timeout;
    }
  }
  const char* request_timeout = getenv("KINESIS_REQUEST_TIMEOUT_MSEC");
  if (request_timeout) {
    int64 timeout;

    if (strings::safe_strto64(request_timeout, &timeout)) {
      config.requestTimeoutMs = timeout;
    }
  }

  return &config;
}

Aws::Client::ClientConfiguration& GetDefaultClientConfig() {
  static Aws::Client::ClientConfiguration* config =
      InitializeDefaultClientConfig();
  return *config;
}

static mutex mu(LINKER_INITIALIZED);
static unsigned count(0);
void AwsInitAPI() {
  mutex_lock lock(mu);
  count++;
  if (count == 1) {
    Aws::SDKOptions options;
    options.cryptoOptions.sha256Factory_create_fn = []() {
      return Aws::MakeShared<AWSSHA256Factory>(AWSCryptoAllocationTag);
    };
    options.cryptoOptions.sha256HMACFactory_create_fn = []() {
      return Aws::MakeShared<AWSSHA256HmacFactory>(AWSCryptoAllocationTag);
    };
    Aws::InitAPI(options);
  }
}
void AwsShutdownAPI() {
  mutex_lock lock(mu);
  count--;
  if (count == 0) {
    Aws::SDKOptions options;
    Aws::ShutdownAPI(options);
  }
}
void ShutdownClient(Aws::Kinesis::KinesisClient* client) {
  if (client != nullptr) {
    delete client;
    AwsShutdownAPI();
  }
}

class KinesisReadableResource : public ResourceBase {
 public:
  KinesisReadableResource(Env* env)
      : env_(env), client_(nullptr, ShutdownClient), interval_(100000) {}
  virtual ~KinesisReadableResource() {}

  Status Init(const string& input, const std::vector<string>& metadata) {
    mutex_lock l(mu_);

    stream_ = input;
    shard_ = "";
    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find("shard=") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid configuration: ",
                                         metadata[i]);
        }
        shard_ = parts[1];
      }
    }

    AwsInitAPI();
    client_.reset(new Aws::Kinesis::KinesisClient(GetDefaultClientConfig()));

    Aws::Kinesis::Model::DescribeStreamRequest request;
    auto outcome =
        client_->DescribeStream(request.WithStreamName(stream_.c_str()));
    if (!outcome.IsSuccess()) {
      return errors::Unknown(outcome.GetError().GetExceptionName(), ": ",
                             outcome.GetError().GetMessage());
    }
    Aws::String shard;
    Aws::String sequence;
    if (shard_ == "") {
      if (outcome.GetResult().GetStreamDescription().GetShards().size() != 1) {
        return errors::InvalidArgument(
            "shard has to be provided unless the stream only have one "
            "shard, there are ",
            outcome.GetResult().GetStreamDescription().GetShards().size(),
            " shards in stream ", stream_);
      }
      shard = outcome.GetResult()
                  .GetStreamDescription()
                  .GetShards()[0]
                  .GetShardId();
      sequence = outcome.GetResult()
                     .GetStreamDescription()
                     .GetShards()[0]
                     .GetSequenceNumberRange()
                     .GetStartingSequenceNumber();
    } else {
      for (const auto& entry :
           outcome.GetResult().GetStreamDescription().GetShards()) {
        if (entry.GetShardId() == shard_.c_str()) {
          shard = entry.GetShardId();
          sequence = entry.GetSequenceNumberRange().GetStartingSequenceNumber();
          break;
        }
      }
      if (shard == "") {
        return errors::InvalidArgument("no shard ", shard_, " in stream ",
                                       stream_);
      }
    }

    Aws::Kinesis::Model::GetShardIteratorRequest iterator_request;
    auto iterator_outcome = client_->GetShardIterator(
        iterator_request.WithStreamName(stream_.c_str())
            .WithShardId(shard)
            .WithShardIteratorType(
                Aws::Kinesis::Model::ShardIteratorType::AT_SEQUENCE_NUMBER)
            .WithStartingSequenceNumber(sequence));
    if (!iterator_outcome.IsSuccess()) {
      return errors::Unknown(iterator_outcome.GetError().GetExceptionName(),
                             ": ", iterator_outcome.GetError().GetMessage());
    }
    iterator_ = iterator_outcome.GetResult().GetShardIterator();
    return Status::OK();
  }
  Status Read(
      std::function<Status(const TensorShape& shape, Tensor** timestamp_tensor,
                           Tensor** data_tensor, Tensor** partition_tensor,
                           Tensor** sequence_tensor)>
          allocate_func) {
    mutex_lock l(mu_);
    do {
      Aws::Kinesis::Model::GetRecordsRequest request;
      auto outcome = client_->GetRecords(
          request.WithShardIterator(iterator_).WithLimit(1));
      if (!outcome.IsSuccess()) {
        return errors::Unknown(outcome.GetError().GetExceptionName(), ": ",
                               outcome.GetError().GetMessage());
      }
      if (outcome.GetResult().GetRecords().size() == 0) {
        // TODO: break if option provided, as no records were returned then
        // nothing is available at the moment.

        // Otherwise continue the loop after a period of time.
        env_->SleepForMicroseconds(interval_);
        continue;
      }
      if (outcome.GetResult().GetRecords().size() != 1) {
        return errors::Unknown("invalid number of records ",
                               outcome.GetResult().GetRecords().size(),
                               " returned");
      }
      iterator_ = outcome.GetResult().GetNextShardIterator();

      Tensor* timestamp_tensor;
      Tensor* data_tensor;
      Tensor* partition_tensor;
      Tensor* sequence_tensor;
      TF_RETURN_IF_ERROR(allocate_func(TensorShape({1}), &timestamp_tensor,
                                       &data_tensor, &partition_tensor,
                                       &sequence_tensor));
      const auto& timestamp =
          outcome.GetResult().GetRecords()[0].GetApproximateArrivalTimestamp();
      const auto& data = outcome.GetResult().GetRecords()[0].GetData();
      const auto& partition =
          outcome.GetResult().GetRecords()[0].GetPartitionKey();
      const auto& sequence =
          outcome.GetResult().GetRecords()[0].GetSequenceNumber();
      timestamp_tensor->flat<int64>()(0) = timestamp.Millis();
      data_tensor->flat<tstring>()(0) =
          string(reinterpret_cast<const char*>(data.GetUnderlyingData()),
                 data.GetLength());
      partition_tensor->flat<tstring>()(0) =
          string(partition.c_str(), partition.size());
      sequence_tensor->flat<tstring>()(0) =
          string(sequence.c_str(), sequence.size());
      return Status::OK();
    } while (true);
    return Status::OK();
  }
  string DebugString() const override {
    mutex_lock l(mu_);
    return "KinesisReadableResource";
  }

 protected:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
  string stream_ GUARDED_BY(mu_);
  string shard_ GUARDED_BY(mu_);
  Aws::String iterator_ GUARDED_BY(mu_);
  std::unique_ptr<Aws::Kinesis::KinesisClient, decltype(&ShutdownClient)>
      client_ GUARDED_BY(mu_);
  int64 interval_ GUARDED_BY(mu_);
};

class KinesisReadableInitOp : public ResourceOpKernel<KinesisReadableResource> {
 public:
  explicit KinesisReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<KinesisReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<KinesisReadableResource>::Compute(context);

    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string& input = input_tensor->scalar<tstring>()();

    std::vector<string> metadata;
    const Tensor* metadata_tensor;
    OP_REQUIRES_OK(context, context->input("metadata", &metadata_tensor));
    for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
      metadata.push_back(metadata_tensor->flat<tstring>()(i));
    }

    OP_REQUIRES_OK(context, resource_->Init(input, metadata));
  }
  Status CreateResource(KinesisReadableResource** resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new KinesisReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};

class KinesisReadableReadOp : public OpKernel {
 public:
  explicit KinesisReadableReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    KinesisReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    OP_REQUIRES_OK(
        context,
        resource->Read([&](const TensorShape& shape, Tensor** timestamp_tensor,
                           Tensor** data_tensor, Tensor** partition_tensor,
                           Tensor** sequence_tensor) -> Status {
          TF_RETURN_IF_ERROR(
              context->allocate_output(0, shape, timestamp_tensor));
          TF_RETURN_IF_ERROR(context->allocate_output(1, shape, data_tensor));
          TF_RETURN_IF_ERROR(
              context->allocate_output(2, shape, partition_tensor));
          TF_RETURN_IF_ERROR(
              context->allocate_output(3, shape, sequence_tensor));
          return Status::OK();
        }));
  }

 private:
  mutable mutex mu_;
  Env* env_ GUARDED_BY(mu_);
};
REGISTER_KERNEL_BUILDER(Name("IO>KinesisReadableInit").Device(DEVICE_CPU),
                        KinesisReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IO>KinesisReadableRead").Device(DEVICE_CPU),
                        KinesisReadableReadOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
