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

#include "tensorflow_io/ignite/kernels/ggfs/ggfs_client.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

GGFSClient::GGFSClient(string host, int32 port, string username, string password, string certfile, string keyfile, string cert_password) {
	LOG(INFO) << "Call GGFSClient::GGFSClient [host = " << host << ", port = " << port << ", username = " << username << ", certfile = " << certfile << ", keyfile = " << keyfile << "]";
}

GGFSClient::~GGFSClient() {
	LOG(INFO) << "Call GGFSClient::~GGFSClient";
}

Status GGFSClient::WriteFile(string path, bool create, bool append, uint8_t* data, int32_t length) {
	LOG(INFO) << "Call GGFSClient::WriteFile [path = " << path << ", create = " << create << ", append = " << append << "]";
	
  TF_RETURN_IF_ERROR(SendCommonRequestHeader(kWriteFileMethodId, 12 + path.length() + length));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(reinterpret_cast<const uint8_t*>(path.c_str()), path.length()));
  TF_RETURN_IF_ERROR(client_->WriteByte(create));
  TF_RETURN_IF_ERROR(client_->WriteByte(append));
  TF_RETURN_IF_ERROR(client_->WriteByte(kByteArrayVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(length));
  TF_RETURN_IF_ERROR(client_->WriteData(data, length));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  return Status::OK();
}

Status GGFSClient::ReadFile(string path, uint8_t** out_data, int32_t* out_length) {
	LOG(INFO) << "Call GGFSClient::ReadFile [path = " << path << "]";
	
  TF_RETURN_IF_ERROR(SendCommonRequestHeader(kReadFileMethodId, 5 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(reinterpret_cast<const uint8_t*>(path.c_str()), path.length()));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  return Status::OK();
}

Status GGFSClient::Move(string from, string to) {
	LOG(INFO) << "Call GGFSClient::Move [from = " << from << ", to = " << to << "]";
	
  TF_RETURN_IF_ERROR(SendCommonRequestHeader(kMoveMethodId, 10 + from.length() + to.length()));
  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(from.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(reinterpret_cast<const uint8_t*>(from.c_str()), from.length()));
  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(to.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(reinterpret_cast<const uint8_t*>(to.c_str()), to.length()));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  return Status::OK();
}

Status GGFSClient::Stat(string path, bool* out_is_directory, int64_t* out_modification_time, int32_t* out_size) {
	LOG(INFO) << "Call GGFSClient::Stat [path = " << path << "]";
	
  TF_RETURN_IF_ERROR(SendCommonRequestHeader(kStatMethodId, 5 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(reinterpret_cast<const uint8_t*>(path.c_str()), path.length()));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  return Status::OK();
}

Status GGFSClient::Exists(string path) {
	LOG(INFO) << "Call GGFSClient::Exists [path = " << path << "]";

  TF_RETURN_IF_ERROR(SendCommonRequestHeader(kExistsMethodId, 5 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(reinterpret_cast<const uint8_t*>(path.c_str()), path.length()));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  uint8_t res;
  TF_RETURN_IF_ERROR(client_->ReadByte(&res));

  if (!res)
    return errors::NotFound("File ", path, " not found");

  return Status::OK();
}

Status GGFSClient::Remove(string path) {
	LOG(INFO) << "Call GGFSClient::Remove [path = " << path << "]";

  TF_RETURN_IF_ERROR(SendCommonRequestHeader(kRemoveMethodId, 5 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(reinterpret_cast<const uint8_t*>(path.c_str()), path.length()));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

	return Status::OK();
}

Status GGFSClient::MkDir(string path, bool only_if_not_exists) {
	LOG(INFO) << "Call GGFSClient::MkDir [path = " << path << ", only_if_not_exists = " << only_if_not_exists << "]";
	
  TF_RETURN_IF_ERROR(SendCommonRequestHeader(kMkDirMethodId, 6 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(reinterpret_cast<const uint8_t*>(path.c_str()), path.length()));
  TF_RETURN_IF_ERROR(client_->WriteByte(only_if_not_exists));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  return Status::OK();
}

Status GGFSClient::MkDirs(string path, bool only_if_not_exists) {
	LOG(INFO) << "Call GGFSClient::MkDirs [path = " << path << ", only_if_not_exists = " << only_if_not_exists << "]";
	
  TF_RETURN_IF_ERROR(SendCommonRequestHeader(kMkDirsMethodId, 6 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(reinterpret_cast<const uint8_t*>(path.c_str()), path.length()));
  TF_RETURN_IF_ERROR(client_->WriteByte(only_if_not_exists));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  return Status::OK();
}

Status GGFSClient::ListFiles(string path, std::vector<string>* out_files) {
	LOG(INFO) << "Call GGFSClient::ListFiles [path = " << path << "]";
	
  TF_RETURN_IF_ERROR(SendCommonRequestHeader(kListFilesMethodId, 5 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(reinterpret_cast<const uint8_t*>(path.c_str()), path.length()));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  return Status::OK();
}

Status GGFSClient::SendCommonRequestHeader(uint8_t method_id, int32_t length) {
  TF_RETURN_IF_ERROR(client_->WriteInt(32 + length));
  TF_RETURN_IF_ERROR(client_->WriteShort(kCustomProcessorOpcode));
  TF_RETURN_IF_ERROR(client_->WriteLong(0));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(16));
  TF_RETURN_IF_ERROR(client_->WriteData(reinterpret_cast<const uint8_t*>("ML_MODEL_STORAGE"), 16));

  TF_RETURN_IF_ERROR(client_->WriteByte(method_id));

  return Status::OK();
}

Status GGFSClient::ReceiveCommonResponseHeader() {
  int32_t res_len;
  TF_RETURN_IF_ERROR(client_->ReadInt(&res_len));

  int64_t req_id;
  TF_RETURN_IF_ERROR(client_->ReadLong(&req_id));

  int32_t status;
  TF_RETURN_IF_ERROR(client_->ReadInt(&status));

  if (status != 0) {
    uint8_t err_msg_header;
    TF_RETURN_IF_ERROR(client_->ReadByte(&err_msg_header));
    if (err_msg_header == kStringVal) {
      int32_t err_msg_length;
      TF_RETURN_IF_ERROR(client_->ReadInt(&err_msg_length));

      uint8_t* err_msg_c = new uint8_t[err_msg_length];
      auto clean = gtl::MakeCleanup([err_msg_c] { delete[] err_msg_c; });
      TF_RETURN_IF_ERROR(client_->ReadData(err_msg_c, err_msg_length));
      string err_msg(reinterpret_cast<char*>(err_msg_c), err_msg_length);

      return errors::Unknown("Close Resource Error [status=", status,
                               ", message=", err_msg, "]");
    }
    return errors::Unknown("Close Resource Error [status=", status, "]");
  }

  return Status::OK();
}

Status GGFSClient::EstablishConnection() {
  if (!client_->IsConnected()) {
    TF_RETURN_IF_ERROR(client_->Connect());

    Status status = Handshake();
    if (!status.ok()) {
      Status disconnect_status = client_->Disconnect();
      if (!disconnect_status.ok()) LOG(ERROR) << disconnect_status.ToString();

      return status;
    }
  }

  return Status::OK();
}

Status GGFSClient::Handshake() {
  int32_t msg_len = kHandshakeReqDefaultLength;

  if (username_.empty())
    msg_len += 1;
  else
    msg_len += 5 + username_.length();  // 1 byte header, 4 bytes length.

  if (password_.empty())
    msg_len += 1;
  else
    msg_len += 5 + password_.length();  // 1 byte header, 4 bytes length.

  TF_RETURN_IF_ERROR(client_->WriteInt(msg_len));
  TF_RETURN_IF_ERROR(client_->WriteByte(1));
  TF_RETURN_IF_ERROR(client_->WriteShort(kProtocolMajorVersion));
  TF_RETURN_IF_ERROR(client_->WriteShort(kProtocolMinorVersion));
  TF_RETURN_IF_ERROR(client_->WriteShort(kProtocolPatchVersion));
  TF_RETURN_IF_ERROR(client_->WriteByte(2));
  if (username_.empty()) {
    TF_RETURN_IF_ERROR(client_->WriteByte(kNullVal));
  } else {
    TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
    TF_RETURN_IF_ERROR(client_->WriteInt(username_.length()));
    TF_RETURN_IF_ERROR(
        client_->WriteData(reinterpret_cast<const uint8_t*>(username_.c_str()),
                           username_.length()));
  }

  if (password_.empty()) {
    TF_RETURN_IF_ERROR(client_->WriteByte(kNullVal));
  } else {
    TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
    TF_RETURN_IF_ERROR(client_->WriteInt(password_.length()));
    TF_RETURN_IF_ERROR(
        client_->WriteData(reinterpret_cast<const uint8_t*>(password_.c_str()),
                           password_.length()));
  }

  int32_t handshake_res_len;
  TF_RETURN_IF_ERROR(client_->ReadInt(&handshake_res_len));
  uint8_t handshake_res;
  TF_RETURN_IF_ERROR(client_->ReadByte(&handshake_res));

  if (handshake_res != 1) {
    int16_t serv_ver_major;
    TF_RETURN_IF_ERROR(client_->ReadShort(&serv_ver_major));
    int16_t serv_ver_minor;
    TF_RETURN_IF_ERROR(client_->ReadShort(&serv_ver_minor));
    int16_t serv_ver_patch;
    TF_RETURN_IF_ERROR(client_->ReadShort(&serv_ver_patch));
    uint8_t header;
    TF_RETURN_IF_ERROR(client_->ReadByte(&header));

    if (header == kStringVal) {
      int32_t length;
      TF_RETURN_IF_ERROR(client_->ReadInt(&length));

      uint8_t* err_msg_c = new uint8_t[length];
      auto clean = gtl::MakeCleanup([err_msg_c] { delete[] err_msg_c; });
      TF_RETURN_IF_ERROR(client_->ReadData(err_msg_c, length));
      string err_msg(reinterpret_cast<char*>(err_msg_c), length);

      return errors::Unknown("Handshake Error [result=", handshake_res,
                             ", version=", serv_ver_major, ".", serv_ver_minor,
                             ".", serv_ver_patch, ", message='", err_msg, "']");
    } else if (header == kNullVal) {
      return errors::Unknown("Handshake Error [result=", handshake_res,
                             ", version=", serv_ver_major, ".", serv_ver_minor,
                             ".", serv_ver_patch, "]");
    } else {
      return errors::Unknown("Handshake Error [result=", handshake_res,
                             ", version=", serv_ver_major, ".", serv_ver_minor,
                             ".", serv_ver_patch, "]");
    }
  }

  return Status::OK();
}

}  // namespace tensorflow