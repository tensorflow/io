/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_SPARSE_VALUE_BUFFER_H_
#define TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_SPARSE_VALUE_BUFFER_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_io/core/kernels/avro/atds/errors.h"

namespace tensorflow {
namespace atds {
namespace sparse {

template <typename T>
using vecvec = std::vector<std::vector<T>>;

struct ValueBuffer {
  vecvec<int> int_values;
  vecvec<long> long_values;
  vecvec<float> float_values;
  vecvec<double> double_values;
  vecvec<bool> bool_values;
  vecvec<string> string_values;

  vecvec<long> indices;
  vecvec<size_t> num_of_elements;
};

template <typename T>
std::vector<T>& GetValueVector(ValueBuffer& buffer, size_t index);

template <>
inline std::vector<int>& GetValueVector(ValueBuffer& buffer, size_t index) {
  return buffer.int_values[index];
}

template <>
inline std::vector<long>& GetValueVector(ValueBuffer& buffer, size_t index) {
  return buffer.long_values[index];
}

template <>
inline std::vector<float>& GetValueVector(ValueBuffer& buffer, size_t index) {
  return buffer.float_values[index];
}

template <>
inline std::vector<double>& GetValueVector(ValueBuffer& buffer, size_t index) {
  return buffer.double_values[index];
}

template <>
inline std::vector<string>& GetValueVector(ValueBuffer& buffer, size_t index) {
  return buffer.string_values[index];
}

template <>
inline std::vector<bool>& GetValueVector(ValueBuffer& buffer, size_t index) {
  return buffer.bool_values[index];
}

template <typename T>
const std::vector<T>& GetValueVector(const ValueBuffer& buffer, size_t index);

template <>
inline const std::vector<int>& GetValueVector(const ValueBuffer& buffer,
                                              size_t index) {
  return buffer.int_values[index];
}

template <>
inline const std::vector<long>& GetValueVector(const ValueBuffer& buffer,
                                               size_t index) {
  return buffer.long_values[index];
}

template <>
inline const std::vector<float>& GetValueVector(const ValueBuffer& buffer,
                                                size_t index) {
  return buffer.float_values[index];
}

template <>
inline const std::vector<double>& GetValueVector(const ValueBuffer& buffer,
                                                 size_t index) {
  return buffer.double_values[index];
}

template <>
inline const std::vector<string>& GetValueVector(const ValueBuffer& buffer,
                                                 size_t index) {
  return buffer.string_values[index];
}

template <>
inline const std::vector<bool>& GetValueVector(const ValueBuffer& buffer,
                                               size_t index) {
  return buffer.bool_values[index];
}

inline Status FillIndicesTensor(const std::vector<long>& buffer, Tensor& tensor,
                                size_t offset) {
  void* dest =
      reinterpret_cast<void*>(reinterpret_cast<long*>(tensor.data()) + offset);
  const void* src = reinterpret_cast<const void*>(buffer.data());
  size_t len = buffer.size() * sizeof(long);
  std::memcpy(dest, src, len);
  return OkStatus();
}

template <typename T>
inline Status FillValuesTensor(const sparse::ValueBuffer& buffer,
                               Tensor& tensor, size_t values_index,
                               size_t offset) {
  auto& values = GetValueVector<T>(buffer, values_index);
  void* dest =
      reinterpret_cast<void*>(reinterpret_cast<T*>(tensor.data()) + offset);
  const void* src = reinterpret_cast<const void*>(values.data());
  size_t len = values.size() * sizeof(T);
  std::memcpy(dest, src, len);
  return OkStatus();
}

template <>
inline Status FillValuesTensor<string>(const sparse::ValueBuffer& buffer,
                                       Tensor& tensor, size_t values_index,
                                       size_t offset) {
  auto& values = buffer.string_values[values_index];
  for (size_t i = 0; i < values.size(); i++) {
    tensor.flat<tstring>()(offset++) = std::move(values[i]);
  }
  return OkStatus();
}

template <>
inline Status FillValuesTensor<bool>(const sparse::ValueBuffer& buffer,
                                     Tensor& tensor, size_t values_index,
                                     size_t offset) {
  auto& values = buffer.bool_values[values_index];
  for (size_t i = 0; i < values.size(); i++) {
    tensor.flat<bool>()(offset++) = values[i];
  }
  return OkStatus();
}

inline Status FillValuesTensor(const sparse::ValueBuffer& buffer,
                               Tensor& values_tensor, DataType dtype,
                               size_t values_index, size_t offset) {
  switch (dtype) {
    case DT_INT32: {
      return FillValuesTensor<int>(buffer, values_tensor, values_index, offset);
    }
    case DT_INT64: {
      return FillValuesTensor<long>(buffer, values_tensor, values_index,
                                    offset);
    }
    case DT_FLOAT: {
      return FillValuesTensor<float>(buffer, values_tensor, values_index,
                                     offset);
    }
    case DT_DOUBLE: {
      return FillValuesTensor<double>(buffer, values_tensor, values_index,
                                      offset);
    }
    case DT_STRING: {
      return FillValuesTensor<string>(buffer, values_tensor, values_index,
                                      offset);
    }
    case DT_BOOL: {
      return FillValuesTensor<bool>(buffer, values_tensor, values_index,
                                    offset);
    }
    default: {
      return TypeNotSupportedError(dtype);
    }
  }
}

}  // namespace sparse
}  // namespace atds
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_AVRO_ATDS_SPARSE_VALUE_BUFFER_H_
