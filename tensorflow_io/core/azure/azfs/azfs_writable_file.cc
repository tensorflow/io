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

#include "tensorflow_io/core/azure/azfs/azfs_writable_file.h"
#include "tensorflow_io/core/azure/azfs/azfs_client.h"

namespace tensorflow {
namespace io {

// TODO: DO NOT use a hardcoded path
Status GetTmpFilename(std::string *filename) {
  if (!filename) {
    return errors::Internal("'filename' cannot be nullptr.");
  }
#ifndef _WIN32
  char buffer[] = "/tmp/az_blob_filesystem_XXXXXX";
  int fd = mkstemp(buffer);
  if (fd < 0) {
    return errors::Internal("Failed to create a temporary file.");
  }
#else
  char buffer[] = "/tmp/az_blob_filesystem_XXXXXX";
  char *ret = _mktemp(buffer);
  if (ret == nullptr) {
    return errors::Internal("Failed to create a temporary file.");
  }
#endif
  *filename = buffer;
  return Status::OK();
}

AzBlobWritableFile::AzBlobWritableFile(const std::string &account,
                                       const std::string &container,
                                       const std::string &object)
    : account_(account),
      container_(container),
      object_(object),
      sync_needed_(true) {
  if (GetTmpFilename(&tmp_content_filename_).ok()) {
    outfile_.open(tmp_content_filename_,
                  std::ofstream::binary | std::ofstream::app);
  }
}

AzBlobWritableFile::~AzBlobWritableFile() { Close().IgnoreError(); }

struct BlockMetadata {
  uint64 start;
  uint64 end;
  std::string id;
};

tensorflow::Status AzBlobWritableFile::Append(tensorflow::StringPiece data) {
  TF_RETURN_IF_ERROR(CheckWritable());
  sync_needed_ = true;
  outfile_ << data;
  if (!outfile_.good()) {
    return errors::Internal("Could not append to the internal temporary file.");
  }
  return Status::OK();
}

Status AzBlobWritableFile::Close() {
  if (outfile_.is_open()) {
    TF_RETURN_IF_ERROR(Sync());
    outfile_.close();
    std::remove(tmp_content_filename_.c_str());
  }
  return Status::OK();
}

Status AzBlobWritableFile::Flush() { return Sync(); }

Status AzBlobWritableFile::Sync() {
  TF_RETURN_IF_ERROR(CheckWritable());
  if (!sync_needed_) {
    return Status::OK();
  }
  const auto status = SyncImpl();
  if (status.ok()) {
    sync_needed_ = false;
  }
  return status;
}

Status AzBlobWritableFile::SyncImpl() {
  outfile_.flush();
  if (!outfile_.good()) {
    return errors::Internal("Could not write to the internal temporary file.");
  }

  auto blob_client = CreateAzBlobClientWrapper(account_);
  blob_client.upload_file_to_blob(tmp_content_filename_, container_, object_);
  if (errno != 0) {
    return errors::Internal("Failed to upload to az://", account_, "/",
                            container_, "/", object_, " (", errno_to_string(),
                            ")");
  }

  return Status::OK();
}

Status AzBlobWritableFile::CheckWritable() const {
  if (!outfile_.is_open()) {
    return errors::FailedPrecondition(
        "The internal temporary file is not writable.");
  }
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow
