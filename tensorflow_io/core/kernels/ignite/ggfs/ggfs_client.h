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

#ifndef TENSORFLOW_CONTRIB_IGNITE_KERNELS_GGFS_GGFS_CLIENT_H_
#define TENSORFLOW_CONTRIB_IGNITE_KERNELS_GGFS_GGFS_CLIENT_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_io/core/kernels/ignite/client/ignite_client.h"

namespace tensorflow {

class GGFSClient {
 public:
  GGFSClient(const string &host, const int32_t port, const string &username,
             const string &password, const string &certfile,
             const string &keyfile, const string &cert_password);

  Status WriteFile(const string &path, const bool create, const bool append,
                   const uint8_t *data, const int32_t length);
  Status ReadFile(const string &path, std::shared_ptr<uint8_t> *out_data,
                  int32_t *out_length);
  Status Move(const string &from, const string &to);
  Status Stat(const string &path, bool *out_is_directory,
              int64_t *out_modification_time, int32_t *out_size);
  Status Exists(const string &path);
  Status Remove(const string &path);
  Status MkDir(const string &path, const bool only_if_not_exists);
  Status MkDirs(const string &path, const bool only_if_not_exists);
  Status ListFiles(const string &path, std::vector<string> *out_files);
  string MakeRelative(const string &a, const string &b);

 private:
  std::shared_ptr<Client> client_;

  const string username_;
  const string password_;

  Status EstablishConnection();
  Status Handshake();

  Status SendCommonRequestHeader(uint8_t method_id, int32_t length);
  Status ReceiveCommonResponseHeader();
};

constexpr uint8_t kWriteFileMethodId = 0;
constexpr uint8_t kReadFileMethodId = 1;
constexpr uint8_t kMoveMethodId = 2;
constexpr uint8_t kStatMethodId = 3;
constexpr uint8_t kExistsMethodId = 4;
constexpr uint8_t kRemoveMethodId = 5;
constexpr uint8_t kMkDirMethodId = 6;
constexpr uint8_t kMkDirsMethodId = 7;
constexpr uint8_t kListFilesMethodId = 8;

constexpr uint8_t kNullVal = 101;
constexpr uint8_t kStringVal = 9;
constexpr uint8_t kByteArrayVal = 12;
constexpr uint8_t kProtocolMajorVersion = 1;
constexpr uint8_t kProtocolMinorVersion = 1;
constexpr uint8_t kProtocolPatchVersion = 0;
constexpr int16_t kCustomProcessorOpcode = 32000;
constexpr int16_t kCloseConnectionOpcode = 0;
constexpr int32_t kCloseConnectionReqLength = 18;
constexpr int32_t kHandshakeReqDefaultLength = 8;
constexpr int32_t kMinResLength = 12;

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IGNITE_KERNELS_GGFS_GGFS_CLIENT_H_
