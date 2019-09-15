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

#include <openssl/hmac.h>
#include <openssl/sha.h>

#include <aws/core/Aws.h>
#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/core/utils/crypto/Factories.h>
#include <aws/core/utils/crypto/HMAC.h>
#include <aws/core/utils/crypto/Hash.h>
#include <aws/core/utils/crypto/HashResult.h>
#include <aws/core/utils/Outcome.h>
#include <aws/kinesis/KinesisClient.h>
#include <aws/kinesis/model/DescribeStreamRequest.h>
#include <aws/kinesis/model/GetRecordsRequest.h>
#include <aws/kinesis/model/GetShardIteratorRequest.h>
#include <aws/kinesis/model/PutRecordsRequest.h>
#include <aws/kinesis/model/ShardIteratorType.h>
#include "tensorflow_io/kinesis/kernels/aws_kernels.h"

namespace tensorflow {
namespace data {

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

}  // namespace data
}  // namespace tensorflow
