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

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_io/arrow/kernels/arrow_stream_client.h"
#include "tensorflow_io/arrow/kernels/arrow_util.h"

namespace tensorflow {
namespace data {

ArrowStreamClient::ArrowStreamClient(const std::string& endpoint)
    : endpoint_(endpoint), sock_(-1), pos_(0) {}

ArrowStreamClient::~ArrowStreamClient() {
  if (sock_ != -1) {
    Close();
  }
}

arrow::Status ArrowStreamClient::Connect() {
  string socket_family;
  string host;
  Status status;

  status = ArrowUtil::ParseEndpoint(endpoint_, &socket_family, &host);
  if (!status.ok()) {
    return arrow::Status::Invalid("Error parsing endpoint string: " +
                                  endpoint_);
  }

  if (socket_family.empty() || socket_family == "tcp") {
    std::string addr_str;
    std::string port_str;
    status = ArrowUtil::ParseHost(host, &addr_str, &port_str);
    if (!status.ok()) {
      return arrow::Status::Invalid("Error parsing host string: " + host);
    }

    int port_num = std::stoi(port_str);
    struct sockaddr_in serv_addr;

    if (sock_ == -1) {
      if ((sock_ = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        return arrow::Status::IOError("Socket creation error");
      }
    }

    bzero((char*)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_addr.s_addr = inet_addr(addr_str.c_str());
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port_num);

    if (connect(sock_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
      return arrow::Status::IOError("Connection failed to AF_INET: " + host);
    }

  } else if (socket_family == "unix") {
    if (sock_ == -1) {
      if ((sock_ = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
        return arrow::Status::IOError("Socket creation error");
      }
    }

    struct sockaddr_un serv_addr;
    bzero((char*)&serv_addr, sizeof(serv_addr));
    serv_addr.sun_family = AF_UNIX;
    strcpy(serv_addr.sun_path, host.c_str());

    if (connect(sock_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
      return arrow::Status::IOError("Connection failed to AF_UNIX: " + host);
    }

  } else {
    return arrow::Status::Invalid("Unsupported socket family: " +
                                  socket_family);
  }

  return arrow::Status::OK();
}

arrow::Status ArrowStreamClient::Close() {
  int status = close(sock_);
  sock_ = -1;

  if (status != 0) {
    return arrow::Status::IOError("Failed to correctly close connection");
  }

  return arrow::Status::OK();
}

bool ArrowStreamClient::closed() const { return sock_ == -1; }

arrow::Result<int64_t> ArrowStreamClient::Tell() const { return pos_; }

arrow::Result<int64_t> ArrowStreamClient::Read(int64_t nbytes, void* out) {
  // TODO: 0 bytes requested when message body length == 0
  if (nbytes == 0) {
    return 0;
  }

  int status = recv(sock_, out, nbytes, MSG_WAITALL);
  if (status == 0) {
    return arrow::Status::IOError("connection closed unexpectedly");
  } else if (status < 0) {
    return arrow::Status::IOError("error reading from socket");
  }

  pos_ += nbytes;
  return nbytes;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> ArrowStreamClient::Read(
    int64_t nbytes) {
  arrow::Result<std::shared_ptr<arrow::ResizableBuffer>> result =
      arrow::AllocateResizableBuffer(nbytes);
  ARROW_RETURN_NOT_OK(result);
  std::shared_ptr<arrow::ResizableBuffer> buffer =
      std::move(result).ValueUnsafe();
  int64_t bytes_read;
  ARROW_ASSIGN_OR_RAISE(bytes_read, Read(nbytes, buffer->mutable_data()));
  ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read, false));
  buffer->ZeroPadding();
  return buffer;
}

}  // namespace data
}  // namespace tensorflow
