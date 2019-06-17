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

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow_io/ignite/kernels/client/ignite_plain_client.h"
#include "tensorflow_io/ignite/kernels/client/ignite_ssl_wrapper.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_io/ignite/kernels/ggfs/ggfs.h"
#include "tensorflow_io/ignite/kernels/ggfs/ggfs_random_access_file.h"
#include "tensorflow_io/ignite/kernels/ggfs/ggfs_writable_file.h"

namespace tensorflow {

std::unique_ptr<GGFSClient> GGFS::CreateClient() const {
  return std::unique_ptr<GGFSClient>(new GGFSClient("localhost", 10800, "", "", "", "", ""));
}

GGFS::GGFS() {
  LOG(INFO) << "Call GGFS::GGFS";
}

GGFS::~GGFS() {
  LOG(INFO) << "Call GGFS::~GGFS";
}

Status GGFS::NewRandomAccessFile(const string &file_name,
                                 std::unique_ptr<RandomAccessFile> *result) {
  LOG(INFO) << "Call GGFS::NewRandomAccessFile [file_name = " << file_name << "]";

  std::unique_ptr<GGFSClient> client = CreateClient();
  result->reset(new GGFSRandomAccessFile(TranslateName(file_name), std::move(client)));

  return Status::OK();
}

Status GGFS::NewWritableFile(const string &file_name,
                             std::unique_ptr<WritableFile> *result) {
  LOG(INFO) << "Call GGFS::NewWritableFile [file_name = " << file_name << "]";
  
  std::unique_ptr<GGFSClient> client = CreateClient();
  result->reset(new GGFSWritableFile(TranslateName(file_name), std::move(client)));

  return Status::OK();
}

Status GGFS::NewAppendableFile(const string &file_name,
                               std::unique_ptr<WritableFile> *result) {
  LOG(INFO) << "Call GGFS::NewAppendableFile [file_name = " << file_name << "]";
  
  std::unique_ptr<GGFSClient> client = CreateClient();
  result->reset(new GGFSWritableFile(TranslateName(file_name), std::move(client)));
  
  return Status::OK();
}

Status GGFS::NewReadOnlyMemoryRegionFromFile(
    const string &file_name, std::unique_ptr<ReadOnlyMemoryRegion> *result) {
  LOG(INFO) << "Call GGFS::NewReadOnlyMemoryRegionFromFile [file_name = " << file_name << "]";
  
  return Status::OK();
}

Status GGFS::FileExists(const string &file_name) {
  LOG(INFO) << "Call GGFS::FileExists [file_name = " << file_name << "]";

  return Status::OK();
}

Status GGFS::GetChildren(const string &file_name, std::vector<string> *result) {
  LOG(INFO) << "Call GGFS::GetChildren [file_name = " << file_name << "]";
  
  std::unique_ptr<GGFSClient> client = CreateClient();
  return client->ListFiles(TranslateName(file_name), result);
}

Status GGFS::GetMatchingPaths(const string &pattern,
                              std::vector<string> *results) {
  LOG(INFO) << "Call GGFS::GetMatchingPaths [pattern = " << pattern << "]";

  return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
}

Status GGFS::DeleteFile(const string &file_name) {
  LOG(INFO) << "Call GGFS::DeleteFile [file_name = " << file_name << "]";
  
  std::unique_ptr<GGFSClient> client = CreateClient();
  return client->Remove(TranslateName(file_name));
}

Status GGFS::CreateDir(const string &file_name) {
  LOG(INFO) << "Call GGFS::CreateDir [file_name = " << file_name << "]";
  
  std::unique_ptr<GGFSClient> client = CreateClient();
  return client->MkDirs(TranslateName(file_name), false);
}

Status GGFS::DeleteDir(const string &file_name) {
  LOG(INFO) << "Call GGFS::DeleteDir [file_name = " << file_name << "]";

  std::unique_ptr<GGFSClient> client = CreateClient();
  return client->Remove(TranslateName(file_name));
}

Status GGFS::GetFileSize(const string &file_name, uint64 *size) {
  LOG(INFO) << "Call GGFS::GetFileSize [file_name = " << file_name << "]";
  
  bool is_directory;
  int64_t modification_time;

  std::unique_ptr<GGFSClient> client = CreateClient();
  return client->Stat(TranslateName(file_name), &is_directory, &modification_time, (int32_t *) size);
}

Status GGFS::RenameFile(const string &src, const string &dst) {
  LOG(INFO) << "Call GGFS::RenameFile [src = " << src << ", dst = " << dst << "]";
  
  std::unique_ptr<GGFSClient> client = CreateClient();
  return client->Move(TranslateName(src), TranslateName(dst));
}

Status GGFS::Stat(const string &file_name, FileStatistics *stats) {
  LOG(INFO) << "Call GGFS::Stat [file_name = " << file_name << "]";
  
  bool is_directory;
  int64_t modification_time;
  int32_t size;

  std::unique_ptr<GGFSClient> client = CreateClient();
  TF_RETURN_IF_ERROR(client->Stat(TranslateName(file_name), &is_directory, &modification_time, &size));

  *stats = FileStatistics(size, modification_time * 1000000, is_directory);

  return Status::OK();
}

string GGFS::TranslateName(const string &name) const {
  LOG(INFO) << "Call GGFS::TranslateName [name = " << name << "]";

  StringPiece scheme, namenode, path;
  io::ParseURI(name, &scheme, &namenode, &path);
  return string(path.data(), path.length());
}

}  // namespace tensorflow
