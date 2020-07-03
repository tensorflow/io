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

#include "tensorflow_io/core/kernels/ignite/ggfs/ggfs.h"

#include <queue>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow_io/core/kernels/ignite/client/ignite_plain_client.h"
#include "tensorflow_io/core/kernels/ignite/client/ignite_ssl_wrapper.h"
#include "tensorflow_io/core/kernels/ignite/ggfs/ggfs_random_access_file.h"
#include "tensorflow_io/core/kernels/ignite/ggfs/ggfs_writable_file.h"

namespace tensorflow {

Status GGFS::UpdateConnectionProperties() {
  const char *env_host = std::getenv("IGNITE_HOST");
  const char *env_port = std::getenv("IGNITE_PORT");
  const char *env_username = std::getenv("IGNITE_USERNAME");
  const char *env_password = std::getenv("IGNITE_PASSWORD");
  const char *env_certfile = std::getenv("IGNITE_CERTFILE");
  const char *env_keyfile = std::getenv("IGNITE_KEYFILE");
  const char *env_cert_password = std::getenv("IGNITE_CERT_PASSWORD");

  if (env_host) host = string(env_host);

  if (env_port && !strings::safe_strto32(env_port, &port)) {
    return errors::Unknown(
        "IGNITE_PORT environment variable is not a valid integer: ", env_port);
  }

  if (env_username) username = string(env_username);

  if (env_password) password = string(env_password);

  if (env_certfile) certfile = string(env_certfile);

  if (env_keyfile) keyfile = string(env_keyfile);

  if (env_cert_password) cert_password = string(env_cert_password);

  return Status::OK();
}

Status GGFS::NewRandomAccessFile(const string &file_name,
                                 std::unique_ptr<RandomAccessFile> *result) {
  LOG(INFO) << "Call GGFS::NewRandomAccessFile [file_name = " << file_name
            << "]";

  TF_RETURN_IF_ERROR(UpdateConnectionProperties());

  result->reset(new GGFSRandomAccessFile(
      TranslateName(file_name),
      std::unique_ptr<GGFSClient>(new GGFSClient(
          host, port, username, password, certfile, keyfile, cert_password))));

  return Status::OK();
}

Status GGFS::NewWritableFile(const string &file_name,
                             std::unique_ptr<WritableFile> *result) {
  LOG(INFO) << "Call GGFS::NewWritableFile [file_name = " << file_name << "]";

  TF_RETURN_IF_ERROR(UpdateConnectionProperties());

  result->reset(new GGFSWritableFile(
      TranslateName(file_name),
      std::unique_ptr<GGFSClient>(new GGFSClient(
          host, port, username, password, certfile, keyfile, cert_password))));

  return Status::OK();
}

Status GGFS::NewAppendableFile(const string &file_name,
                               std::unique_ptr<WritableFile> *result) {
  LOG(INFO) << "Call GGFS::NewAppendableFile [file_name = " << file_name << "]";

  TF_RETURN_IF_ERROR(UpdateConnectionProperties());

  result->reset(new GGFSWritableFile(
      TranslateName(file_name),
      std::unique_ptr<GGFSClient>(new GGFSClient(
          host, port, username, password, certfile, keyfile, cert_password))));

  return Status::OK();
}

Status GGFS::NewReadOnlyMemoryRegionFromFile(
    const string &file_name, std::unique_ptr<ReadOnlyMemoryRegion> *result) {
  LOG(INFO) << "Call GGFS::NewReadOnlyMemoryRegionFromFile [file_name = "
            << file_name << "]";

  return errors::Unimplemented("GGFS does not support ReadOnlyMemoryRegion");
}

Status GGFS::FileExists(const string &file_name) {
  LOG(INFO) << "Call GGFS::FileExists [file_name = " << file_name << "]";

  TF_RETURN_IF_ERROR(UpdateConnectionProperties());

  GGFSClient client = {host,     port,    username,     password,
                       certfile, keyfile, cert_password};
  return client.Exists(TranslateName(file_name));
}

Status GGFS::GetChildren(const string &file_name, std::vector<string> *result) {
  LOG(INFO) << "Call GGFS::GetChildren [file_name = " << file_name << "]";

  TF_RETURN_IF_ERROR(UpdateConnectionProperties());

  GGFSClient client = {host,     port,    username,     password,
                       certfile, keyfile, cert_password};
  return client.ListFiles(TranslateName(file_name), result);
}

Status GGFS::GetMatchingPaths(const string &pattern,
                              std::vector<string> *results) {
  LOG(INFO) << "Call GGFS::GetMatchingPaths [pattern = " << pattern << "]";
#if defined(_MSC_VER)
  return errors::Unimplemented("GGFS::GetMatchingPaths");
#else
  return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
#endif
}

Status GGFS::DeleteFile(const string &file_name) {
  LOG(INFO) << "Call GGFS::DeleteFile [file_name = " << file_name << "]";

  TF_RETURN_IF_ERROR(UpdateConnectionProperties());

  GGFSClient client = {host,     port,    username,     password,
                       certfile, keyfile, cert_password};
  return client.Remove(TranslateName(file_name));
}

Status GGFS::CreateDir(const string &file_name) {
  LOG(INFO) << "Call GGFS::CreateDir [file_name = " << file_name << "]";

  TF_RETURN_IF_ERROR(UpdateConnectionProperties());

  GGFSClient client = {host,     port,    username,     password,
                       certfile, keyfile, cert_password};
  return client.MkDirs(TranslateName(file_name), false);
}

Status GGFS::DeleteDir(const string &file_name) {
  LOG(INFO) << "Call GGFS::DeleteDir [file_name = " << file_name << "]";

  TF_RETURN_IF_ERROR(UpdateConnectionProperties());

  GGFSClient client = {host,     port,    username,     password,
                       certfile, keyfile, cert_password};
  return client.Remove(TranslateName(file_name));
}

Status GGFS::GetFileSize(const string &file_name, uint64 *size) {
  LOG(INFO) << "Call GGFS::GetFileSize [file_name = " << file_name << "]";

  bool is_directory;
  int64_t modification_time;

  TF_RETURN_IF_ERROR(UpdateConnectionProperties());

  GGFSClient client = {host,     port,    username,     password,
                       certfile, keyfile, cert_password};
  return client.Stat(TranslateName(file_name), &is_directory,
                     &modification_time, (int32_t *)size);
}

Status GGFS::RenameFile(const string &src, const string &dst) {
  LOG(INFO) << "Call GGFS::RenameFile [src = " << src << ", dst = " << dst
            << "]";

  TF_RETURN_IF_ERROR(UpdateConnectionProperties());

  GGFSClient client = {host,     port,    username,     password,
                       certfile, keyfile, cert_password};

  bool is_directory;
  int64_t modification_time;
  int32_t size;

  TF_RETURN_IF_ERROR(client.Stat(TranslateName(src), &is_directory,
                                 &modification_time, &size));

  if (!is_directory) return client.Move(TranslateName(src), TranslateName(dst));

  std::queue<string> file_queue;
  std::queue<string> dir_queue;
  dir_queue.push(src);

  TF_RETURN_IF_ERROR(client.Remove(TranslateName(dst)));

  while (!dir_queue.empty()) {
    string src_dir = dir_queue.front();
    dir_queue.pop();
    string dst_dir = dst + "/" + client.MakeRelative(src_dir, src);

    TF_RETURN_IF_ERROR(client.MkDirs(TranslateName(dst_dir), false));

    std::vector<string> children;
    TF_RETURN_IF_ERROR(client.ListFiles(TranslateName(src_dir), &children));

    for (string const &child : children) {
      string full_child_path = src_dir + "/" + child;
      TF_RETURN_IF_ERROR(client.Stat(TranslateName(full_child_path),
                                     &is_directory, &modification_time, &size));
      std::queue<string> target_queue = is_directory ? dir_queue : file_queue;
      target_queue.push(full_child_path);
    }
  }

  TF_RETURN_IF_ERROR(client.Remove(TranslateName(src)));

  return Status::OK();
}

Status GGFS::Stat(const string &file_name, FileStatistics *stats) {
  LOG(INFO) << "Call GGFS::Stat [file_name = " << file_name << "]";

  bool is_directory;
  int64_t modification_time;
  int32_t size;

  TF_RETURN_IF_ERROR(UpdateConnectionProperties());

  GGFSClient client = {host,     port,    username,     password,
                       certfile, keyfile, cert_password};
  TF_RETURN_IF_ERROR(client.Stat(TranslateName(file_name), &is_directory,
                                 &modification_time, &size));

  *stats = FileStatistics(size, modification_time * 1000000, is_directory);

  return Status::OK();
}

string GGFS::TranslateName(const string &name) const {
  LOG(INFO) << "Call GGFS::TranslateName [name = " << name << "]";

  StringPiece scheme, namenode, path;
  io::ParseURI(name, &scheme, &namenode, &path);

  string res = string(path.data(), path.length());

  if (res.length() != 0 && res.at(res.length() - 1) == '/')
    res = res.substr(0, res.length() - 1);

  return res;
}

REGISTER_FILE_SYSTEM("ggfs", GGFS);

}  // namespace tensorflow
