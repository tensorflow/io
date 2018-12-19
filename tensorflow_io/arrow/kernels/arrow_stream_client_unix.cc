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

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "tensorflow_io/arrow/kernels/arrow_stream_client.h"

namespace tensorflow {

ArrowStreamClient::ArrowStreamClient(const std::string& host)
    : host_(host), sock_(-1), pos_(0) {}

ArrowStreamClient::~ArrowStreamClient() {
  if (sock_ != -1) {
    Close();
  }
}

arrow::Status ArrowStreamClient::Connect() {
  size_t sep_pos = host_.find(':');
  if (sep_pos == std::string::npos || sep_pos == host_.size()) {
    return arrow::Status::Invalid(
        "Expected host to be in format <host>:<port> but got: " + host_);
  }
  std::string host_str = host_.substr(0, sep_pos);
  std::string port_str = host_.substr(sep_pos + 1, host_.size() - sep_pos);
  int port_num = std::stoi(port_str);
  struct sockaddr_in serv_addr;

  if (sock_ == -1) {
    if ((sock_ = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      return arrow::Status::IOError("Socket creation error");
    }
  }

  bzero((char*)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_addr.s_addr = inet_addr(host_str.c_str());
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port_num);

  if (connect(sock_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
    return arrow::Status::IOError("Connection failed to host: " + host_);
  }

  return arrow::Status::OK();
}

arrow::Status ArrowStreamClient::Close() {
  int status = close(sock_);
  sock_ = 1;

  if (status != 0) {
    return arrow::Status::IOError("Failed to correctly close connection");
  }

  return arrow::Status::OK();
}

arrow::Status ArrowStreamClient::Tell(int64_t* position) const {
  *position = pos_;
  return arrow::Status::OK();
}

arrow::Status ArrowStreamClient::Read(int64_t nbytes,
                                      int64_t* bytes_read,
                                      void* out) {
  // TODO: look into why 0 bytes are requested
  if (nbytes == 0) {
    return arrow::Status::OK();
  }

  int status = recv(sock_, out, nbytes, MSG_WAITALL);
  if (status == 0) {
    return arrow::Status::IOError("connection closed unexpectedly");
  } else if (status < 0) {
    return arrow::Status::IOError("error reading from socket");
  }

  *bytes_read = nbytes;
  pos_ += *bytes_read;

  return arrow::Status::OK();
}

arrow::Status ArrowStreamClient::Read(int64_t nbytes,
                                      std::shared_ptr<arrow::Buffer>* out) {
  std::shared_ptr<arrow::ResizableBuffer> buffer;
  ARROW_RETURN_NOT_OK(arrow::AllocateResizableBuffer(nbytes, &buffer));
  int64_t bytes_read;
  ARROW_RETURN_NOT_OK(Read(nbytes, &bytes_read, buffer->mutable_data()));
  ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read, false));
  buffer->ZeroPadding();
  *out = buffer;
  return arrow::Status::OK();
}

}  // namespace tensorflow
