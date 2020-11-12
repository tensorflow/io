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
#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_S3_AWS_CRYPTO_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_S3_AWS_CRYPTO_H_

#include <aws/core/Aws.h>
#include <aws/core/utils/crypto/Factories.h>
#include <aws/core/utils/crypto/HMAC.h>
#include <aws/core/utils/crypto/Hash.h>
#include <aws/core/utils/crypto/SecureRandom.h>

namespace tensorflow {
namespace io {
namespace s3 {
namespace tf_s3_filesystem {
constexpr char AWSCryptoAllocationTag[] = "AWSCryptoAllocation";

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

class AWSSecureRandomFactory : public Aws::Utils::Crypto::SecureRandomFactory {
 public:
  std::shared_ptr<Aws::Utils::Crypto::SecureRandomBytes> CreateImplementation()
      const override;
};

}  // namespace tf_s3_filesystem
}  // namespace s3
}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_S3_AWS_CRYPTO_H_
