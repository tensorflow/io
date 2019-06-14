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

#include "tensorflow_io/ignite/kernels/client/ignite_client.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class GGFSClient {
 public:
  GGFSClient(string host, int32 port, string username, string password, string certfile, string keyfile, string cert_password);
  ~GGFSClient();

  Status WriteFile(string path, bool create, bool append, uint8_t* data, int32_t length);
  Status ReadFile(string path, uint8_t** out_data, int32_t* out_length);
  Status Move(string from, string to);
  Status Stat(string path, bool* out_is_directory, int64_t* out_modification_time, int32_t* out_size);
  Status Exists(string path);
  Status Remove(string path);
  Status MkDir(string path, bool only_if_not_exists);
  Status MkDirs(string path, bool only_if_not_exists);
  Status ListFiles(string path, std::vector<string>* out_files);

 private:
  std::unique_ptr<Client> client_;

  const string username_;
  const string password_;

  Status EstablishConnection();
  Status Handshake();

  Status SendCommonRequestHeader(uint8_t method_id, int32_t length);
  Status ReceiveCommonResponseHeader();
};

constexpr uint8_t kWriteFileMethodId = 0;
constexpr uint8_t kReadFileMethodId = 0;
constexpr uint8_t kMoveMethodId = 0;
constexpr uint8_t kStatMethodId = 0;
constexpr uint8_t kExistsMethodId = 0;
constexpr uint8_t kRemoveMethodId = 0;
constexpr uint8_t kMkDirMethodId = 0;
constexpr uint8_t kMkDirsMethodId = 0;
constexpr uint8_t kListFilesMethodId = 0;

constexpr uint8_t kNullVal = 101;
constexpr uint8_t kStringVal = 9;
constexpr uint8_t kByteArrayVal = 12;
constexpr uint8_t kProtocolMajorVersion = 1;
constexpr uint8_t kProtocolMinorVersion = 1;
constexpr uint8_t kProtocolPatchVersion = 0;
constexpr int16_t kCustomProcessorOpcode = 32000;
constexpr int16_t kScanQueryOpcode = 2000;
constexpr int16_t kLoadNextPageOpcode = 2001;
constexpr int16_t kCloseConnectionOpcode = 0;
constexpr int32_t kScanQueryReqLength = 25;
constexpr int32_t kScanQueryResHeaderLength = 25;
constexpr int32_t kLoadNextPageReqLength = 18;
constexpr int32_t kLoadNextPageResHeaderLength = 17;
constexpr int32_t kCloseConnectionReqLength = 18;
constexpr int32_t kHandshakeReqDefaultLength = 8;
constexpr int32_t kMinResLength = 12;

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IGNITE_KERNELS_GGFS_GGFS_CLIENT_H_