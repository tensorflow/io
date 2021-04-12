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

#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")
#pragma comment(lib, "Mswsock.lib")
#pragma comment(lib, "AdvApi32.lib")

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
  if (socket_family == "unix") {
    return arrow::Status::Invalid("Unsupported socket family: " +
                                  socket_family);
  }

  string addr_str;
  string port_str;
  status = ArrowUtil::ParseHost(host, &addr_str, &port_str);
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

  res = getaddrinfo(addr_str.c_str(), port_str.c_str(), &hints, &result);
  if (res != 0) {
    return arrow::Status::IOError("Getaddrinfo failed with error: ",
                                  std::to_string(res));
  }

  std::unique_ptr<addrinfo, void (*)(addrinfo*)> result_scope(
      result, [](addrinfo* p) {
        if (p != nullptr) {
          freeaddrinfo(p);
        }
      });

  for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {
    sock_ = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
    if (sock_ == INVALID_SOCKET) {
      WSACleanup();
      return arrow::Status::IOError("Socket failed with error: ",
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

bool ArrowStreamClient::closed() const { return sock_ == -1; }

arrow::Result<int64_t> ArrowStreamClient::Tell() const { return pos_; }

arrow::Result<int64_t> ArrowStreamClient::Read(int64_t nbytes, void* out) {
  // TODO: look into why 0 bytes are requested
  if (nbytes == 0) {
    return 0;
  }

  int status = recv(sock_, (char*)out, nbytes, 0);
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
