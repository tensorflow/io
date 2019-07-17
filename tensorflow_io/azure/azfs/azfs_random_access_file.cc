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

#include <sstream>
#include <string>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow_io/azure/azfs/azfs_client.h"
#include "tensorflow_io/azure/azfs/azfs_random_access_file.h"

namespace tensorflow {

AzBlobRandomAccessFile::AzBlobRandomAccessFile(const std::string &account,
                                               const std::string &container,
                                               const std::string &object)
    : account_(account), container_(container), object_(object) {}

Status AzBlobRandomAccessFile::Read(uint64 offset, size_t n,
                                    StringPiece *result, char *scratch) const {
  auto blob_client = CreateAzBlobClientWrapper(account_);

  std::ostringstream oss;
  // https://stackoverflow.com/a/12481580
#ifndef __APPLE__
  oss.rdbuf()->pubsetbuf(scratch, n);
#endif

  blob_client.download_blob_to_stream(container_, object_, offset, n, oss);
  if (errno != 0) {
    n = 0;
    *result = StringPiece("", 0);
    return errors::Internal("Failed to get contents of az://", account_,
                            kAzBlobEndpoint, "/", container_, "/", object_,
                            " (", errno_to_string(), ")");
  }

#ifndef __APPLE__
  *result = StringPiece(scratch, n);
#else
  auto blob_string = oss.str();
  if (scratch != nullptr) {
    std::copy(blob_string.begin(), blob_string.end(), scratch);
  }
  *result = StringPiece(blob_string);
#endif

  return Status::OK();
}
}  // namespace tensorflow
