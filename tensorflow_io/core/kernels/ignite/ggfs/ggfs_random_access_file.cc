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

#include "tensorflow_io/core/kernels/ignite/ggfs/ggfs_random_access_file.h"

namespace tensorflow {

GGFSRandomAccessFile::GGFSRandomAccessFile(const string &file_name,
                                           std::unique_ptr<GGFSClient> &&client)
    : file_name_(file_name), client_(std::move(client)) {}

GGFSRandomAccessFile::~GGFSRandomAccessFile() {}

Status GGFSRandomAccessFile::Read(uint64 offset, size_t n, StringPiece *result,
                                  char *scratch) const {
  std::shared_ptr<uint8_t> data = {};
  int32_t length;

  TF_RETURN_IF_ERROR(client_->ReadFile(file_name_, &data, &length));

  if (offset >= length) return errors::OutOfRange("End of file");

  int32_t size = n < length - offset ? n : length - offset;
  std::copy(data.get() + offset, data.get() + offset + size, scratch);
  *result = StringPiece(scratch, size);

  return Status::OK();
}

}  // namespace tensorflow
