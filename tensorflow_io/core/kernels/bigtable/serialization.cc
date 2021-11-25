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

#include "rpc/xdr.h"
#include "rpc/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace io {

inline StatusOr<float> BytesToFloat(std::string const& s) {
  float v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_float(&xdrs, &v)) {
    return errors::InvalidArgument("Error reading float from byte array.");
  }
  return v;
}

inline StatusOr<double> BytesToDouble(std::string const& s) {
  double v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_double(&xdrs, &v)) {
    return errors::InvalidArgument("Error reading double from byte array.");
  }
  return v;
}

inline StatusOr<int64_t> BytesToInt64(std::string const& s) {
  int64_t v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_int64_t(&xdrs, &v)) {
    return errors::InvalidArgument("Error reading int64 from byte array.");
  }
  return v;
}

inline StatusOr<int32_t> BytesToInt32(std::string const& s) {
  int32_t v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_int32_t(&xdrs, &v)) {
    return errors::InvalidArgument("Error reading int32 from byte array.");
  }
  return v;
}

inline StatusOr<bool_t> BytesToBool(std::string const& s) {
  bool_t v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_bool(&xdrs, &v)) {
    return errors::InvalidArgument("Error reading bool from byte array.");
  }
  return v;
}

Status PutCellValueInTensor(Tensor& tensor, size_t index, DataType cell_type,
                            google::cloud::bigtable::Cell const& cell) {
  switch (cell_type) {
    case DT_STRING: {
      auto tensor_data = tensor.tensor<tstring, 1>();
      tensor_data(index) = std::string(cell.value());
    } break;
    case DT_BOOL: {
      auto tensor_data = tensor.tensor<bool, 1>();
      auto maybe_parsed_data = BytesToBool(cell.value());
      if (!maybe_parsed_data.ok()) {
        return maybe_parsed_data.status();
      }
      tensor_data(index) = maybe_parsed_data.ValueOrDie();
    } break;
    case DT_INT32: {
      auto tensor_data = tensor.tensor<int32_t, 1>();
      auto maybe_parsed_data = BytesToInt32(cell.value());
      if (!maybe_parsed_data.ok()) {
        return maybe_parsed_data.status();
      }
      tensor_data(index) = maybe_parsed_data.ValueOrDie();
    } break;
    case DT_INT64: {
      auto tensor_data = tensor.tensor<int64_t, 1>();
      auto maybe_parsed_data = BytesToInt64(cell.value());
      if (!maybe_parsed_data.ok()) {
        return maybe_parsed_data.status();
      }
      tensor_data(index) = maybe_parsed_data.ValueOrDie();
    } break;
    case DT_FLOAT: {
      auto tensor_data = tensor.tensor<float, 1>();
      auto maybe_parsed_data = BytesToFloat(cell.value());
      if (!maybe_parsed_data.ok()) {
        return maybe_parsed_data.status();
      }
      tensor_data(index) = maybe_parsed_data.ValueOrDie();
    } break;
    case DT_DOUBLE: {
      auto tensor_data = tensor.tensor<double, 1>();
      auto maybe_parsed_data = BytesToDouble(cell.value());
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
