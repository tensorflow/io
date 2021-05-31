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

#ifndef TENSORFLOW_IO_CORE_KERNELS_ARROW_UTIL_H_
#define TENSORFLOW_IO_CORE_KERNELS_ARROW_UTIL_H_

#include "arrow/api.h"
#include "arrow/ipc/api.h"
#include "arrow/util/io_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Forward declaration
class Tensor;
class TensorShape;

namespace data {

#define CHECK_ARROW(arrow_status)             \
  do {                                        \
    arrow::Status _s = (arrow_status);        \
    if (!_s.ok()) {                           \
      return errors::Internal(_s.ToString()); \
    }                                         \
  } while (false)

namespace ArrowUtil {

// Convert Arrow Data Type to TensorFlow
Status GetTensorFlowType(std::shared_ptr<::arrow::DataType> dtype,
                         ::tensorflow::DataType* out);

// Convert TensorFlow Data Type to Arrow
Status GetArrowType(::tensorflow::DataType dtype,
                    std::shared_ptr<::arrow::DataType>* out);

// Assign equivalent TensorShape for the given Arrow Array
Status AssignShape(std::shared_ptr<arrow::Array> array, int64 i,
                   int64 batch_size, TensorShape* out_shape);

// Assign DataType and equivalent TensorShape for the given Arrow Array
Status AssignSpec(std::shared_ptr<arrow::Array> array, int64 i,
                  int64 batch_size, ::tensorflow::DataType* out_dtype,
                  TensorShape* out_shape);

// Assign elements of an Arrow Array to a Tensor
Status AssignTensor(std::shared_ptr<arrow::Array> array, int64 i,
                    Tensor* out_tensor);

// Checks the Arrow Array datatype matches the expected TF datatype
Status CheckArrayType(std::shared_ptr<arrow::DataType> type,
                      ::tensorflow::DataType expected_type);

// Make list and primitive array data
Status MakeArrayData(std::shared_ptr<arrow::DataType> type,
                     std::vector<int64> array_lengths,
                     std::vector<std::shared_ptr<arrow::Buffer>> buffers,
                     std::shared_ptr<arrow::ArrayData>* out_data);

// Parse the given endpoint to extract type and value strings
Status ParseEndpoint(std::string endpoint, std::string* endpoint_type,
                     std::string* endpoint_value);

// Parse the given IPv4 host string to get address and port
Status ParseHost(std::string host, std::string* host_address,
                 std::string* host_port);

}  // namespace ArrowUtil
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_CORE_KERNELS_ARROW_UTIL_H_
