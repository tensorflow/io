/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

#include "tensorflow_io/core/kernels/bigtable/serialization.h"

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"

namespace cbt = ::google::cloud::bigtable;

namespace tensorflow {
namespace io {
namespace {

#ifdef _WIN32

#include <winsock.h>

inline StatusOr<int32_t> BytesToInt32(const cbt::Cell& cell) {
  std::string const& bytes = cell.value();
  union {
    char bytes[4];
    int32_t res;
  } u;
  if (bytes.size() != 4U) {
    return errors::InvalidArgument("Invalid int32 representation.");
  }
  memcpy(u.bytes, bytes.data(), 4);
  return ntohl(u.res);
}

inline StatusOr<int64_t> BytesToInt64(const cbt::Cell& cell) {
  auto maybe_value = cell.decode_big_endian_integer<int64_t>();
  if (!maybe_value.ok()) {
    return errors::InvalidArgument("Invalid int32 representation.");
  }
  return maybe_value.value();
}

inline StatusOr<float> BytesToFloat(const cbt::Cell& cell) {
  auto const int_rep = BytesToInt32(cell);
  if (!int_rep.ok()) {
    return int_rep;
  }
  union {
    float res;
    int32_t int_rep;
  } u;
  u.int_rep = *int_rep;
  return u.res;
}

inline StatusOr<double> BytesToDouble(const cbt::Cell& cell) {
  auto const int_rep = BytesToInt64(cell);
  if (!int_rep.ok()) {
    return int_rep;
  }
  union {
    double res;
    int64_t int_rep;
  } u;
  u.int_rep = *int_rep;
  return u.res;
}

#else  // _WIN32

#include "rpc/types.h"
#include "rpc/xdr.h"

inline StatusOr<float> BytesToFloat(const cbt::Cell& cell) {
  std::string const& s = cell.value();
  float v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_float(&xdrs, &v)) {
    return errors::InvalidArgument("Error reading float from byte array.");
  }
  return v;
}

inline StatusOr<double> BytesToDouble(const cbt::Cell& cell) {
  std::string const& s = cell.value();
  double v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_double(&xdrs, &v)) {
    return errors::InvalidArgument("Error reading double from byte array.");
  }
  return v;
}

inline StatusOr<int64_t> BytesToInt64(const cbt::Cell& cell) {
  std::string const& s = cell.value();
  int64_t v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_int64_t(&xdrs, &v)) {
    return errors::InvalidArgument("Error reading int64 from byte array.");
  }
  return v;
}

inline StatusOr<int32_t> BytesToInt32(const cbt::Cell& cell) {
  std::string const& s = cell.value();
  int32_t v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_int32_t(&xdrs, &v)) {
    return errors::InvalidArgument("Error reading int32 from byte array.");
  }
  return v;
}

#endif  // _WIN32

inline StatusOr<bool> BytesToBool(const cbt::Cell& cell) {
  std::string const& bytes = cell.value();
  if (bytes.size() != 1U) {
    return errors::InvalidArgument("Invalid bool representation.");
  }
  return (*bytes.data()) != 0;
}

}  // namespace

Status PutCellValueInTensor(Tensor& tensor, size_t index, DataType cell_type,
                            google::cloud::bigtable::Cell const& cell) {
  switch (cell_type) {
    case DT_STRING: {
      auto tensor_data = tensor.tensor<tstring, 1>();
      tensor_data(index) = std::string(cell.value());
    } break;
    case DT_BOOL: {
      auto tensor_data = tensor.tensor<bool, 1>();
      auto maybe_parsed_data = BytesToBool(cell);
      if (!maybe_parsed_data.ok()) {
        return maybe_parsed_data.status();
      }
      tensor_data(index) = maybe_parsed_data.ValueOrDie();
    } break;
    case DT_INT32: {
      auto tensor_data = tensor.tensor<int32_t, 1>();
      auto maybe_parsed_data = BytesToInt32(cell);
      if (!maybe_parsed_data.ok()) {
        return maybe_parsed_data.status();
      }
      tensor_data(index) = maybe_parsed_data.ValueOrDie();
    } break;
    case DT_INT64: {
      auto tensor_data = tensor.tensor<int64_t, 1>();
      auto maybe_parsed_data = BytesToInt64(cell);
      if (!maybe_parsed_data.ok()) {
        return maybe_parsed_data.status();
      }
      tensor_data(index) = maybe_parsed_data.ValueOrDie();
    } break;
    case DT_FLOAT: {
      auto tensor_data = tensor.tensor<float, 1>();
      auto maybe_parsed_data = BytesToFloat(cell);
      if (!maybe_parsed_data.ok()) {
        return maybe_parsed_data.status();
      }
      tensor_data(index) = maybe_parsed_data.ValueOrDie();
    } break;
    case DT_DOUBLE: {
      auto tensor_data = tensor.tensor<double, 1>();
      auto maybe_parsed_data = BytesToDouble(cell);
      if (!maybe_parsed_data.ok()) {
        return maybe_parsed_data.status();
      }
      tensor_data(index) = maybe_parsed_data.ValueOrDie();
    } break;
    default:
      return errors::Unimplemented("Data type not supported.");
  }
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow
