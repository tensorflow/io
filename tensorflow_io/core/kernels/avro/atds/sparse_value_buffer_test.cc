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

#include "tensorflow_io/core/kernels/avro/atds/sparse_value_buffer.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow_io/core/kernels/avro/atds/decoder_test_util.h"

namespace tensorflow {
namespace atds {
namespace sparse {

class FillIndicesTensorTest : public ::testing::TestWithParam<size_t> {};

TEST_P(FillIndicesTensorTest, Offset) {
  std::vector<long> buffer = {1, 3, 5, 7};
  size_t offset = GetParam();
  int64 limit = static_cast<int64>(buffer.size() + offset);
  Tensor tensor(DT_INT64, {limit});
  Status status = FillIndicesTensor(buffer, tensor, offset);
  ASSERT_TRUE(status.ok());
  AssertTensorRangeEqual(tensor, buffer, offset);
}

INSTANTIATE_TEST_SUITE_P(offset_0_1_2, FillIndicesTensorTest,
                         ::testing::Values(0, 1, 2));

template <typename T>
void FillValuesTensorTest(const std::vector<T>& values, size_t values_index,
                          size_t offset) {
  DataType dtype = GetDataType<T>();

  sparse::ValueBuffer buffer;
  auto& values_buffer = GetValuesBuffer<T>(buffer);
  values_buffer.resize(values_index + 1);
  values_buffer.back() = values;
  int64 size = static_cast<int64>(offset + values.size());
  Tensor tensor(dtype, {size});

  Status status = FillValuesTensor(buffer, tensor, dtype, values_index, offset);
  ASSERT_TRUE(status.ok());
  AssertTensorRangeEqual(tensor, values, offset);
}

TEST(FillValuesTensorTest, DT_INT32) {
  std::vector<int> values = {3, 2, 1, -1};
  FillValuesTensorTest(values, 0, 0);
}

TEST(FillValuesTensorTest, DT_INT64) {
  std::vector<long> values = {-1, -2};
  FillValuesTensorTest(values, 1, 0);
}

TEST(FillValuesTensorTest, DT_FLOAT) {
  std::vector<float> values = {0.0, 1.0, -1.0};
  FillValuesTensorTest(values, 0, 2);
}

TEST(FillValuesTensorTest, DT_DOUBLE) {
  std::vector<double> values = {3.17, 4.02, 5.13};
  FillValuesTensorTest(values, 11, 11);
}

TEST(FillValuesTensorTest, DT_STRING) {
  std::vector<string> values = {"ABC"};
  FillValuesTensorTest(values, 7, 0);
}

TEST(FillValuesTensorTest, DT_BOOL) {
  std::vector<bool> values = {false, true, true};
  FillValuesTensorTest(values, 0, 5);
}

}  // namespace sparse
}  // namespace atds
}  // namespace tensorflow
