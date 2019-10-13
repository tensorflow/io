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

#ifndef TENSORFLOW_IO_AZURE_AZFS_CLIENT_H
#define TENSORFLOW_IO_AZURE_AZFS_CLIENT_H

#include <memory>
#include <string>

#include "blob/blob_client.h"
#include "storage_account.h"
#include "storage_credential.h"
#include "storage_errno.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/file_system.h"

constexpr char kAzBlobEndpoint[] = ".blob.core.windows.net";

namespace tensorflow {
namespace io {

/// \brief Splits a Azure path to a account, container and object.
///
/// For example,
/// "az://account-name.blob.core.windows.net/container/path/to/file.txt" gets
/// split into "account-name", "container" and "path/to/file.txt".
Status ParseAzBlobPath(StringPiece fname, bool empty_object_ok,
                       std::string *account, std::string *container,
                       std::string *object);

std::string errno_to_string();

azure::storage_lite::blob_client_wrapper CreateAzBlobClientWrapper(
    const std::string &account);

}  // namespace io
}  // namespace tensorflow

#endif
