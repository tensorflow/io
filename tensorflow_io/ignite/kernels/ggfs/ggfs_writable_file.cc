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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_io/ignite/kernels/ggfs/ggfs_writable_file.h"

namespace tensorflow {

GGFSWritableFile::GGFSWritableFile(const string &file_name, std::unique_ptr<GGFSClient> &&client)
  : file_name_(file_name),
    client_(std::move(client)) {
  LOG(INFO) << "Call GGFSWritableFile::GGFSWritableFile [file_name = " << file_name_ << "]";
}

GGFSWritableFile::~GGFSWritableFile() {
  LOG(INFO) << "Call GGFSWritableFile::~GGFSWritableFile [file_name = " << file_name_ << "]";
}

Status GGFSWritableFile::Append(StringPiece data) {
  LOG(INFO) << "Call GGFSWritableFile::Append [file_name = " << file_name_ << "]";

  Status exists_status = client_->Exists(file_name_);
  bool create = exists_status.code() == errors::Code::NOT_FOUND;

  return client_->WriteFile(file_name_, create, true, (uint8_t *)data.data(), data.size());
}

Status GGFSWritableFile::Close() {
  LOG(INFO) << "Call GGFSWritableFile::Close [file_name = " << file_name_ << "]";
  
  return Status::OK();
}

Status GGFSWritableFile::Flush() {
  LOG(INFO) << "Call GGFSWritableFile::Flush [file_name = " << file_name_ << "]";

  return Status::OK();
}

Status GGFSWritableFile::Sync() {
  LOG(INFO) << "Call GGFSWritableFile::Sync [file_name = " << file_name_ << "]";

  return Status::OK();
}

}  // namespace tensorflow

