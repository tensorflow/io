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

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")
#pragma comment(lib, "Mswsock.lib")
#pragma comment(lib, "AdvApi32.lib")

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "tensorflow_io/arrow/kernels/arrow_stream_client.h"

namespace tensorflow {

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

  status = ParseEndpoint(endpoint_, &socket_family, &host);
  if (!status.ok()) {
    return arrow::Status::Invalid(
        "Error parsing endpoint string: " + endpoint_);
  }
  if (socket_family == "unix") {
    return arrow::Status::Invalid(
        "Unsupported socket family: " + socket_family);
  }

  string addr_str;
  string port_str;
  status = ParseHost(host, &addr_str, &port_str);
  if (!status.ok()) {
    return arrow::Status::Invalid("Error parsing host string: " + host);
  }

  WSADATA wsaData;
  addrinfo *result = NULL, *ptr = NULL, hints;

  int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
  if (res != 0) {
    return arrow::Status::IOError("WSAStartup failed with error: ",
                                  std::to_string(res));
  }

  ZeroMemory(&hints, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  res = getaddrinfo(addr_str.c_str(), port_str.c_str(), &hints,
                    &result);
  if (res != 0) {
    return arrow::Status::IOError("Getaddrinfo failed with error: ",
                                  std::to_string(res));
  }

  auto clean = gtl::MakeCleanup([result] { freeaddrinfo(result); });

  for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {
    sock_ = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
    if (sock_ == INVALID_SOCKET) {
      WSACleanup();
      return errors::Internal("Socket failed with error: ",
                              std::to_string(WSAGetLastError()));
    }

    res = connect(sock_, ptr->ai_addr, (int)ptr->ai_addrlen);
    if (res == SOCKET_ERROR) {
      closesocket(sock_);
      sock_ = INVALID_SOCKET;
      continue;
    }

    break;
  }

  if (sock_ == INVALID_SOCKET) {
    WSACleanup();
    return arrow::Status::IOError("Unable to connect to server");
  }

  return arrow::Status::OK();
}

arrow::Status ArrowStreamClient::Close() {
  int res = shutdown(sock_, SD_SEND);
  closesocket(sock_);
  WSACleanup();

  if (res == SOCKET_ERROR) {
    return arrow::Status::IOError("Shutdown failed with error: ",
                                  std::to_string(WSAGetLastError()));
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

  int status = recv(sock_, (char *)out, nbytes, 0);
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
