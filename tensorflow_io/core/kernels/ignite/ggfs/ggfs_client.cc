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

#include "tensorflow_io/core/kernels/ignite/ggfs/ggfs_client.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_io/core/kernels/ignite/client/ignite_plain_client.h"
#include "tensorflow_io/core/kernels/ignite/client/ignite_ssl_wrapper.h"

namespace tensorflow {

string GGFSClient::MakeRelative(const string &a, const string &b) {
  if (a == b) return "";

  string max = a;
  string min = b;
  bool first = b.size() > a.size();

  if (first) {
    max = b;
    min = a;
  }

  auto r = mismatch(min.begin(), min.end(), max.begin());
  return string((first ? r.first : r.second), first ? min.end() : max.end());
}

GGFSClient::GGFSClient(const string &host, const int32_t port,
                       const string &username, const string &password,
                       const string &certfile, const string &keyfile,
                       const string &cert_password)
    : username_(username), password_(password) {
  Client *p_client = new PlainClient(std::move(host), port, false);

  if (certfile.empty()) {
    client_ = std::shared_ptr<Client>(p_client);
  } else {
    client_ = std::shared_ptr<Client>(
        new SslWrapper(std::shared_ptr<Client>(p_client), std::move(certfile),
                       std::move(keyfile), std::move(cert_password), false));
  }
}

Status GGFSClient::WriteFile(const string &path, const bool create,
                             const bool append, const uint8_t *data,
                             const int32_t length) {
  TF_RETURN_IF_ERROR(
      SendCommonRequestHeader(kWriteFileMethodId, 12 + path.length() + length));
  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(
      reinterpret_cast<const uint8_t *>(path.c_str()), path.length()));
  TF_RETURN_IF_ERROR(client_->WriteByte(create));
  TF_RETURN_IF_ERROR(client_->WriteByte(append));
  TF_RETURN_IF_ERROR(client_->WriteByte(kByteArrayVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(length));
  TF_RETURN_IF_ERROR(client_->WriteData(data, length));
  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  return Status::OK();
}

Status GGFSClient::ReadFile(const string &path,
                            std::shared_ptr<uint8_t> *out_data,
                            int32_t *out_length) {
  TF_RETURN_IF_ERROR(
      SendCommonRequestHeader(kReadFileMethodId, 5 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(
      reinterpret_cast<const uint8_t *>(path.c_str()), path.length()));
  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  uint8_t type;
  TF_RETURN_IF_ERROR(client_->ReadByte(&type));

  TF_RETURN_IF_ERROR(client_->ReadInt(out_length));

  out_data->reset(new uint8_t[*out_length]);
  TF_RETURN_IF_ERROR(client_->ReadData(out_data->get(), *out_length));

  return Status::OK();
}

Status GGFSClient::Move(const string &from, const string &to) {
  TF_RETURN_IF_ERROR(
      SendCommonRequestHeader(kMoveMethodId, 10 + from.length() + to.length()));
  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(from.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(
      reinterpret_cast<const uint8_t *>(from.c_str()), from.length()));
  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(to.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(
      reinterpret_cast<const uint8_t *>(to.c_str()), to.length()));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  return Status::OK();
}

Status GGFSClient::Stat(const string &path, bool *out_is_directory,
                        int64_t *out_modification_time, int32_t *out_size) {
  TF_RETURN_IF_ERROR(SendCommonRequestHeader(kStatMethodId, 5 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(
      reinterpret_cast<const uint8_t *>(path.c_str()), path.length()));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  uint8_t is_directory;
  TF_RETURN_IF_ERROR(client_->ReadByte(&is_directory));
  *out_is_directory = is_directory != 0;

  TF_RETURN_IF_ERROR(client_->ReadInt(out_size));
  TF_RETURN_IF_ERROR(client_->ReadLong(out_modification_time));

  return Status::OK();
}

Status GGFSClient::Exists(const string &path) {
  TF_RETURN_IF_ERROR(
      SendCommonRequestHeader(kExistsMethodId, 5 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(
      reinterpret_cast<const uint8_t *>(path.c_str()), path.length()));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  uint8_t res;
  TF_RETURN_IF_ERROR(client_->ReadByte(&res));

  if (!res) return errors::NotFound("File ", path, " not found");

  return Status::OK();
}

Status GGFSClient::Remove(const string &path) {
  TF_RETURN_IF_ERROR(
      SendCommonRequestHeader(kRemoveMethodId, 5 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(
      reinterpret_cast<const uint8_t *>(path.c_str()), path.length()));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  return Status::OK();
}

Status GGFSClient::MkDir(const string &path, const bool only_if_not_exists) {
  TF_RETURN_IF_ERROR(
      SendCommonRequestHeader(kMkDirMethodId, 6 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(
      reinterpret_cast<const uint8_t *>(path.c_str()), path.length()));
  TF_RETURN_IF_ERROR(client_->WriteByte(only_if_not_exists));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  return Status::OK();
}

Status GGFSClient::MkDirs(const string &path, const bool only_if_not_exists) {
  TF_RETURN_IF_ERROR(
      SendCommonRequestHeader(kMkDirsMethodId, 6 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(
      reinterpret_cast<const uint8_t *>(path.c_str()), path.length()));
  TF_RETURN_IF_ERROR(client_->WriteByte(only_if_not_exists));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  return Status::OK();
}

Status GGFSClient::ListFiles(const string &path,
                             std::vector<string> *out_files) {
  TF_RETURN_IF_ERROR(
      SendCommonRequestHeader(kListFilesMethodId, 5 + path.length()));

  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(path.length()));
  TF_RETURN_IF_ERROR(client_->WriteData(
      reinterpret_cast<const uint8_t *>(path.c_str()), path.length()));

  TF_RETURN_IF_ERROR(ReceiveCommonResponseHeader());

  int32_t length;
  TF_RETURN_IF_ERROR(client_->ReadInt(&length));

  while (length > 0) {
    uint8_t type;
    TF_RETURN_IF_ERROR(client_->ReadByte(&type));
    if (type != kStringVal)
      return errors::Unknown(
          "Method GGFSClient::ListFiles expects strings in response");

    int32_t str_length;
    TF_RETURN_IF_ERROR(client_->ReadInt(&str_length));

    string str;
    str.resize(str_length);
    TF_RETURN_IF_ERROR(client_->ReadData((uint8_t *)(&str[0]), str_length));

    out_files->push_back(MakeRelative(
        string(reinterpret_cast<const char *>(&str[0]), str_length),
        path + "/"));

    length--;
  }

  return Status::OK();
}

Status GGFSClient::SendCommonRequestHeader(const uint8_t method_id,
                                           const int32_t length) {
  EstablishConnection();
  TF_RETURN_IF_ERROR(client_->WriteInt(32 + length));
  TF_RETURN_IF_ERROR(client_->WriteShort(kCustomProcessorOpcode));
  TF_RETURN_IF_ERROR(client_->WriteLong(0));
  TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
  TF_RETURN_IF_ERROR(client_->WriteInt(16));
  TF_RETURN_IF_ERROR(client_->WriteData(
      reinterpret_cast<const uint8_t *>("ML_MODEL_STORAGE"), 16));
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

      std::unique_ptr<uint8_t[]> err_msg_c =
          std::unique_ptr<uint8_t[]>(new uint8_t[err_msg_length]);
      TF_RETURN_IF_ERROR(client_->ReadData(err_msg_c.get(), err_msg_length));
      string err_msg(reinterpret_cast<char *>(err_msg_c.get()), err_msg_length);

      return errors::Unknown("Error [status=", status, ", message=", err_msg,
                             "]");
    }
    return errors::Unknown("Error [status=", status, "]");
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
        client_->WriteData(reinterpret_cast<const uint8_t *>(username_.c_str()),
                           username_.length()));
  }

  if (password_.empty()) {
    TF_RETURN_IF_ERROR(client_->WriteByte(kNullVal));
  } else {
    TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
    TF_RETURN_IF_ERROR(client_->WriteInt(password_.length()));
    TF_RETURN_IF_ERROR(
        client_->WriteData(reinterpret_cast<const uint8_t *>(password_.c_str()),
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

      std::unique_ptr<uint8_t[]> err_msg_c =
          std::unique_ptr<uint8_t[]>(new uint8_t[length]);
      TF_RETURN_IF_ERROR(client_->ReadData(err_msg_c.get(), length));
      string err_msg(reinterpret_cast<char *>(err_msg_c.get()), length);

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
