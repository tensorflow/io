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

#include "tensorflow_io/core/azure/azfs/azfs_client.h"
#include "logging.h"

namespace tensorflow {
namespace io {

/// \brief Splits a Azure path to a account, container and object.
///
/// For example,
/// "az://account-name.blob.core.windows.net/container/path/to/file.txt" gets
/// split into "account-name", "container" and "path/to/file.txt".
Status ParseAzBlobPath(StringPiece fname, bool empty_object_ok,
                       std::string *account, std::string *container,
                       std::string *object) {
  if (!account || !object) {
    return errors::Internal("account and object cannot be null.");
  }
  StringPiece scheme, accountp, objectp;
  io::ParseURI(fname, &scheme, &accountp, &objectp);
  if (scheme != "az") {
    return errors::InvalidArgument(
        "Azure Blob Storage path doesn't start with 'az://': ", fname);
  }

  // Consume blob.core.windows.net if it exists
  absl::ConsumeSuffix(&accountp, kAzBlobEndpoint);

  if (accountp.empty() || accountp.compare(".") == 0) {
    return errors::InvalidArgument(
        "Azure Blob Storage path doesn't contain a account name: ", fname);
  }

  *account = std::string(accountp);

  absl::ConsumePrefix(&objectp, "/");

  auto pos = objectp.find('/');
  if (pos == std::string::npos) {
    *container = objectp.data();
    *object = "";
  } else {
    *container = std::string(objectp.substr(0, pos));
    *object = std::string(objectp.substr(pos + 1));
  }

  return Status::OK();
}

std::string errno_to_string() {
  switch (errno) {
    /* common errors */
    case invalid_parameters:
      return "invalid_parameters";
    /* client level */
    case client_init_fail:
      return "client_init_fail";
    case client_already_init:
      return "client_already_init";
    case client_not_init:
      return "client_not_init";
    /* container level */
    case container_already_exists:
      return "container_already_exists";
    case container_not_exists:
      return "container_not_exists";
    case container_name_invalid:
      return "container_name_invalid";
    case container_create_fail:
      return "container_create_fail";
    case container_delete_fail:
      return "container_delete_fail";
    /* blob level */
    case blob__already_exists:
      return "blob__already_exists";
    case blob_not_exists:
      return "blob_not_exists";
    case blob_name_invalid:
      return "blob_name_invalid";
    case blob_delete_fail:
      return "blob_delete_fail";
    case blob_list_fail:
      return "blob_list_fail";
    case blob_copy_fail:
      return "blob_copy_fail";
    case blob_no_content_range:
      return "blob_no_content_range";
    /* unknown error */
    case unknown_error:
    default:
      return "unknown_error - " + std::to_string(errno);
  }
}

std::shared_ptr<azure::storage_lite::storage_credential> get_credential(
    const std::string &account) {
  const auto key = std::getenv("TF_AZURE_STORAGE_KEY");
  if (key != nullptr) {
    return std::make_shared<azure::storage_lite::shared_key_credential>(account,
                                                                        key);
  } else {
    return std::make_shared<azure::storage_lite::anonymous_credential>();
  }
}

azure::storage_lite::blob_client_wrapper CreateAzBlobClientWrapper(
    const std::string &account) {
  azure::storage_lite::logger::set_logger(
      [](azure::storage_lite::log_level level, const std::string &log_msg) {
        switch (level) {
          case azure::storage_lite::log_level::info:
            _TF_LOG_INFO << log_msg;
            break;
          case azure::storage_lite::log_level::error:
          case azure::storage_lite::log_level::critical:
            _TF_LOG_ERROR << log_msg;
            break;
          case azure::storage_lite::log_level::warn:
            _TF_LOG_WARNING << log_msg;
            break;
          case azure::storage_lite::log_level::trace:
          case azure::storage_lite::log_level::debug:
          default:
            break;
        }
      });

  const auto use_dev_account = std::getenv("TF_AZURE_USE_DEV_STORAGE");
  if (use_dev_account != nullptr) {
    auto storage_account =
        azure::storage_lite::storage_account::development_storage_account();
    auto blob_client =
        std::make_shared<azure::storage_lite::blob_client>(storage_account, 10);
    azure::storage_lite::blob_client_wrapper blob_client_wrapper(blob_client);
    return blob_client_wrapper;
  }

  const auto use_http_env = std::getenv("TF_AZURE_STORAGE_USE_HTTP");
  const auto use_https = use_http_env == nullptr;
  const auto blob_endpoint =
      std::string(std::getenv("TF_AZURE_STORAGE_BLOB_ENDPOINT") ?: "");

  auto credentials = get_credential(account);
  auto storage_account = std::make_shared<azure::storage_lite::storage_account>(
      account, credentials, use_https, blob_endpoint);
  auto blob_client =
      std::make_shared<azure::storage_lite::blob_client>(storage_account, 10);
  azure::storage_lite::blob_client_wrapper blob_client_wrapper(blob_client);

  return blob_client_wrapper;
}

}  // namespace io
}  // namespace tensorflow
