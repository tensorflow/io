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
#include "gtest/gtest.h"
#include "tensorflow/core/framework/tensor_testutil.h" // tensor equals
#include "tensorflow_io/avro/utils/value_buffer.h"

// Note, these tests do not cover all avro types, because there are enough tests
// in avroc for that.
namespace tensorflow {
namespace data {


// ------------------------------------------------------------
// Test shape builder
// ------------------------------------------------------------
TEST(ShapeBuilderTest, ShapeBuilderEmpty) {
  ShapeBuilder builder;
  size_t dim(builder.GetNumberOfDimensions());
  EXPECT_TRUE(dim == 0);

  TensorShape shape;
  builder.GetDenseShape(&shape);
  EXPECT_EQ(shape.dims(), 0);

  EXPECT_TRUE(builder.HasAllElements(shape));
}

// Note, we built this case into the shape builder to simplify the surrounding logic/state keeping
TEST(ShapeBuilderTest, ShapeBuilderForScalar) {
  ShapeBuilder builder;
  builder.Increment();

  size_t dim(builder.GetNumberOfDimensions());
  EXPECT_EQ(dim, 0);

  TensorShape shape;
  builder.GetDenseShape(&shape);
  EXPECT_EQ(shape.dims(), 0);

  EXPECT_TRUE(builder.HasAllElements(shape));
}

// Test has all elements for single dimension
TEST(ShapeBuilderTest, ShapeBuilderSingleDimension) {
  ShapeBuilder builder;
  builder.BeginMark(); builder.Increment(); builder.Increment(); builder.FinishMark();
  builder.BeginMark(); builder.Increment(); builder.FinishMark();

  size_t dim(builder.GetNumberOfDimensions());
  EXPECT_EQ(dim, 1);

  TensorShape shape;
  builder.GetDenseShape(&shape);
  EXPECT_EQ(shape.dims(), 1);
  EXPECT_EQ(shape.dim_size(0), 2);

  EXPECT_TRUE(!builder.HasAllElements(shape));
}

TEST(ShapeBuilderTest, ShapeBuilderTwoDimensions) {
  ShapeBuilder builder;
  builder.BeginMark();
    builder.BeginMark(); builder.Increment(); builder.FinishMark();
    builder.BeginMark(); builder.FinishMark();
  builder.FinishMark();

  size_t dim(builder.GetNumberOfDimensions());
  EXPECT_EQ(dim, 2);

  TensorShape shape;
  builder.GetDenseShape(&shape);
  EXPECT_EQ(shape.dims(), 2);
  EXPECT_EQ(shape.dim_size(0), 2);
  EXPECT_EQ(shape.dim_size(1), 1);

  EXPECT_TRUE(!builder.HasAllElements(shape));
}

TEST(ShapeBuilderTest, ShapeBuilderManyDimensions) {
  ShapeBuilder builder;
  builder.BeginMark();
    builder.BeginMark();
      builder.BeginMark();
        builder.Increment();
      builder.FinishMark();
      builder.BeginMark();
        builder.Increment();
      builder.FinishMark();
    builder.FinishMark();
    builder.BeginMark();
      builder.BeginMark();
        builder.Increment();
      builder.FinishMark();
    builder.FinishMark();
    builder.BeginMark();
      builder.BeginMark();
      builder.FinishMark();
    builder.FinishMark();
  builder.FinishMark();

  size_t dim(builder.GetNumberOfDimensions());
  EXPECT_EQ(dim, 3);

  TensorShape shape;
  builder.GetDenseShape(&shape);
  EXPECT_EQ(shape.dims(), 3);
  EXPECT_EQ(shape.dim_size(0), 3);
  EXPECT_EQ(shape.dim_size(1), 2);
  EXPECT_EQ(shape.dim_size(2), 1);

  EXPECT_TRUE(!builder.HasAllElements(shape));
}

// ------------------------------------------------------------
// Test Value Buffer -- Simple corner cases
// ------------------------------------------------------------
TEST(ValueBufferTest, BufferCreateAndDestroy) {
  IntValueBuffer buffer;
}


TEST(ValueBufferTest, DenseTensorForEmptyBuffer) {

  IntValueBuffer buffer;
  const TensorShape partial_shape({0}); // user provided shape
  Tensor defaults; // user provided defaults

  TensorShape resolved_shape; // resolved shape for dense tensor
  TF_EXPECT_OK(buffer.ResolveDenseShape(&resolved_shape, partial_shape, defaults.shape()));
  EXPECT_EQ(resolved_shape, partial_shape);

  Tensor tensor(DT_INT32, resolved_shape);
  TF_EXPECT_OK(buffer.MakeDense(&tensor, resolved_shape, defaults));
}

// ------------------------------------------------------------
// Test Value buffer -- Dense
// ------------------------------------------------------------
TEST(ValueBufferTest, Dense1DWithTensorDefault) {

  // Define the shapes
  const TensorShape defaults_shape({3});
  const TensorShape expected_shape(defaults_shape);
  const TensorShape partial_shape(defaults_shape); // what user would define

  // Define the default tensor
  Tensor defaults(DT_INT32, defaults_shape);
  auto defaults_flat = defaults.flat<int>();
  defaults_flat(0) = 11;
  defaults_flat(1) = 12;
  defaults_flat(2) = 13;

  // Create expected tensor
  Tensor tensor_expected(DT_INT32, expected_shape);
  auto tensor_expected_flat = tensor_expected.flat<int>();
  tensor_expected_flat(0) = 0;
  tensor_expected_flat(1) = 1;
  tensor_expected_flat(2) = 13;

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
  buffer.Add(0);
  buffer.Add(1);
  buffer.FinishMark();

  // Get the resolved shape
  TensorShape resolved_shape;
  EXPECT_EQ(buffer.ResolveDenseShape(&resolved_shape, partial_shape, defaults_shape), Status::OK());
  EXPECT_EQ(resolved_shape, expected_shape);

  // Allocate space for the tensor, fill it, and compare it
  Tensor tensor_actual(DT_INT32, resolved_shape);
  EXPECT_EQ(buffer.MakeDense(&tensor_actual, resolved_shape, defaults), Status::OK());
  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}

TEST(ValueBufferTest, Dense1DWithScalarDefault) {

  // Define shapes
  const TensorShape defaults_shape({1});
  const TensorShape expected_shape({3});
  const TensorShape partial_shape(expected_shape); // what the user would define

  // Define the default tensor
  Tensor defaults(DT_INT32, defaults_shape);
  auto defaults_flat = defaults.flat<int>();
  defaults_flat(0) = 100;

  // Create expected tensor
  Tensor tensor_expected(DT_INT32, expected_shape);
  auto tensor_expected_flat = tensor_expected.flat<int>();
  tensor_expected_flat(0) = 1;
  tensor_expected_flat(1) = 100;
  tensor_expected_flat(2) = 100;

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
  buffer.Add(1);
  buffer.FinishMark();

  // Get the resolved shape
  TensorShape resolved_shape;
  EXPECT_EQ(buffer.ResolveDenseShape(&resolved_shape, partial_shape, defaults_shape), Status::OK());
  EXPECT_EQ(resolved_shape, expected_shape);

  // Make the tensor from buffer and compare
  Tensor tensor_actual(DT_INT32, resolved_shape);
  EXPECT_EQ(buffer.MakeDense(&tensor_actual, resolved_shape, defaults), Status::OK());
  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}

// Test when missing a complete inner nested element
// Test when missing elements in the innermost dimension
TEST(ValueBufferTest, Dense3DWithTensorDefault) {

  // Define the default shape
  const TensorShape defaults_shape({2, 3, 4});
  const TensorShape expected_shape(defaults_shape);
  const TensorShape partial_shape(defaults_shape);

  // Define the default tensor
  Tensor defaults(DT_INT32, defaults_shape);
  auto defaults_flat = defaults.flat<int>();
  for (int i_value = 0; i_value < defaults_shape.num_elements(); ++i_value) {
    defaults_flat(i_value) = i_value;
  }

  // Create expected tensor
  Tensor tensor_expected(DT_INT32, expected_shape);
  auto tensor_expected_write = tensor_expected.tensor<int, 3>();
  // 1st block
  tensor_expected_write(0, 0, 0) = 1;
  tensor_expected_write(0, 0, 1) = 2;
  tensor_expected_write(0, 0, 2) = 3;
  tensor_expected_write(0, 0, 3) = 4;
  // 2nd block
  tensor_expected_write(0, 1, 0) = 2;
  tensor_expected_write(0, 1, 1) = 3;
  tensor_expected_write(0, 1, 2) = 4;
  tensor_expected_write(0, 1, 3) = 7; // default value
  // 3rd block
  tensor_expected_write(0, 2, 0) = 8; // default value
  tensor_expected_write(0, 2, 1) = 9; // default value
  tensor_expected_write(0, 2, 2) = 10; // default value
  tensor_expected_write(0, 2, 3) = 11; // default value
  // 4th block
  tensor_expected_write(1, 0, 0) = 5;
  tensor_expected_write(1, 0, 1) = 6;
  tensor_expected_write(1, 0, 2) = 7;
  tensor_expected_write(1, 0, 3) = 15; // default value
  // 5th block
  tensor_expected_write(1, 1, 0) = 8;
  tensor_expected_write(1, 1, 1) = 9;
  tensor_expected_write(1, 1, 2) = 10;
  tensor_expected_write(1, 1, 3) = 11;
  // 6th block
  tensor_expected_write(1, 2, 0) = 12;
  tensor_expected_write(1, 2, 1) = 21; // default value
  tensor_expected_write(1, 2, 2) = 22; // default value
  tensor_expected_write(1, 2, 3) = 23; // default value


  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.BeginMark();
      buffer.BeginMark();
        buffer.Add(1); buffer.Add(2); buffer.Add(3); buffer.Add(4);
      buffer.FinishMark();
      buffer.BeginMark();
        buffer.Add(2); buffer.Add(3); buffer.Add(4); // misses 1
      buffer.FinishMark();
      // misses complete entries for 3rd component
    buffer.FinishMark();
    buffer.BeginMark();
      buffer.BeginMark();
        buffer.Add(5); buffer.Add(6); buffer.Add(7); // misses 1
      buffer.FinishMark();
      buffer.BeginMark();
        buffer.Add(8); buffer.Add(9); buffer.Add(10); buffer.Add(11);
      buffer.FinishMark();
      buffer.BeginMark();
        buffer.Add(12); // misses 3
      buffer.FinishMark();
    buffer.FinishMark();
  buffer.FinishMark();

  // Get the resolved shape
  TensorShape resolved_shape;
  EXPECT_EQ(buffer.ResolveDenseShape(&resolved_shape, partial_shape, defaults_shape), Status::OK());
  EXPECT_EQ(resolved_shape, expected_shape);

  // Make the tensor from buffer
  Tensor tensor_actual(DT_INT32, resolved_shape);
  EXPECT_EQ(buffer.MakeDense(&tensor_actual, resolved_shape, defaults), Status::OK());
  LOG(INFO) << "Tensor defaults: " << defaults.SummarizeValue(24);
  LOG(INFO) << "Tensor actual:   " << tensor_actual.SummarizeValue(24);
  LOG(INFO) << "Tensor expected: " << tensor_expected.SummarizeValue(24);
  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}


// In this test we never have the maximum number of elements for a dimension
// and we miss more then one element in the outer dimensions
TEST(ValueBufferTest, Dense4DWithTensorDefault) {

  // Define the shapes
  const TensorShape defaults_shape({2, 3, 4, 5});
  const TensorShape expected_shape(defaults_shape);
  const TensorShape partial_shape(defaults_shape);

  // Define the default tensor
  Tensor defaults(DT_INT32, defaults_shape);
  auto defaults_flat = defaults.flat<int>();
  for (int i_value = 0; i_value < defaults_shape.num_elements(); ++i_value) {
    defaults_flat(i_value) = i_value;
  }

  // Create expected tensor
  Tensor tensor_expected(DT_INT32, expected_shape);
  // pre-fill with default values
  auto tensor_expected_flat = tensor_expected.flat<int>();
  for (int i_value = 0; i_value < expected_shape.num_elements(); ++i_value) {
    tensor_expected_flat(i_value) = i_value;
  }
  auto tensor_expected_write = tensor_expected.tensor<int, 4>();
  // 1st block
  tensor_expected_write(0, 0, 0, 0) = 1;
  tensor_expected_write(0, 0, 0, 1) = 2;
  tensor_expected_write(0, 0, 0, 2) = 3;
  tensor_expected_write(0, 0, 0, 3) = 4;
  // 2nd block
  tensor_expected_write(0, 0, 1, 0) = 2;
  tensor_expected_write(0, 0, 1, 1) = 3;
  tensor_expected_write(0, 0, 1, 2) = 4;
  // 3rd block
  tensor_expected_write(1, 0, 0, 0) = 5;
  tensor_expected_write(1, 0, 0, 1) = 6;
  tensor_expected_write(1, 0, 0, 2) = 7;
  // 4th block
  tensor_expected_write(1, 0, 1, 0) = 8;
  tensor_expected_write(1, 0, 1, 1) = 9;
  tensor_expected_write(1, 0, 1, 2) = 10;
  tensor_expected_write(1, 0, 1, 3) = 11;
  // 5th block
  tensor_expected_write(1, 0, 2, 0) = 12;

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.BeginMark();
      buffer.BeginMark();
        buffer.BeginMark();
          buffer.Add(1); buffer.Add(2); buffer.Add(3); buffer.Add(4);
        buffer.FinishMark();
        buffer.BeginMark();
          buffer.Add(2); buffer.Add(3); buffer.Add(4);
        buffer.FinishMark();
      buffer.FinishMark();
    buffer.FinishMark();
    buffer.BeginMark();
      buffer.BeginMark();
        buffer.BeginMark();
          buffer.Add(5); buffer.Add(6); buffer.Add(7);
        buffer.FinishMark();
        buffer.BeginMark();
          buffer.Add(8); buffer.Add(9); buffer.Add(10); buffer.Add(11);
        buffer.FinishMark();
        buffer.BeginMark();
          buffer.Add(12);
        buffer.FinishMark();
      buffer.FinishMark();
    buffer.FinishMark();
  buffer.FinishMark();

  // Get the resolved shape
  TensorShape resolved_shape;
  TF_EXPECT_OK(buffer.ResolveDenseShape(&resolved_shape, partial_shape, defaults_shape));
  EXPECT_EQ(resolved_shape, expected_shape);

  // Make the tensor from buffer and compare it
  Tensor tensor_actual(DT_INT32, resolved_shape);
  EXPECT_EQ(buffer.MakeDense(&tensor_actual, resolved_shape, defaults), Status::OK());
  LOG(INFO) << "Tensor defaults: " << defaults.SummarizeValue(24); // display 1st 24
  LOG(INFO) << "Tensor actual:   " << tensor_actual.SummarizeValue(24);
  LOG(INFO) << "Tensor expected: " << tensor_expected.SummarizeValue(24);
  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}

// Test empty dimension
// Test dense 2D with scalar default
TEST(ValueBufferTest, Dense2DWithScalarDefault) {

  // Define the shapes
  const TensorShape defaults_shape({1});
  const TensorShape expected_shape({2, 3});
  const TensorShape partial_shape(expected_shape);

  // Define the default tensor
  Tensor defaults(DT_INT32, defaults_shape);
  auto defaults_flat = defaults.flat<int>();
  defaults_flat(0) = 100;

  // Create expected tensor
  Tensor tensor_expected(DT_INT32, expected_shape);
  auto tensor_expected_write = tensor_expected.tensor<int, 2>();
  tensor_expected_write(0, 0) = 100;
  tensor_expected_write(0, 1) = 100;
  tensor_expected_write(0, 2) = 100;
  tensor_expected_write(1, 0) = 1;
  tensor_expected_write(1, 1) = 2;
  tensor_expected_write(1, 2) = 3;

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.BeginMark();
    buffer.FinishMark();
    buffer.BeginMark();
      buffer.Add(1); buffer.Add(2); buffer.Add(3);
    buffer.FinishMark();
  buffer.FinishMark();

  // Get the resolved shape
  TensorShape resolved_shape;
  EXPECT_EQ(buffer.ResolveDenseShape(&resolved_shape, partial_shape, defaults_shape), Status::OK());
  EXPECT_EQ(resolved_shape, expected_shape);

  // Make the tensor from buffer
  Tensor tensor_actual(DT_INT32, resolved_shape);
  EXPECT_EQ(buffer.MakeDense(&tensor_actual, resolved_shape, defaults), Status::OK());
  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}

// Test with partial tensor shape, where the default provides
// the complete shape information
TEST(ValueBufferTest, ShapeFromDefault) {

  // Define shapes
  const TensorShape defaults_shape({2, 3});
  const TensorShape expected_shape(defaults_shape);
  const PartialTensorShape partial_shape({-1, -1});

  // Define the default tensor
  Tensor defaults(DT_INT32, defaults_shape);
  auto defaults_flat = defaults.flat<int>();
  for (int i_value = 0; i_value < defaults_shape.num_elements(); ++i_value) {
    defaults_flat(i_value) = i_value;
  }

  // Create expected tensor
  Tensor tensor_expected(DT_INT32, expected_shape);
  auto tensor_expected_write = tensor_expected.tensor<int, 2>();
  // 1st block
  tensor_expected_write(0, 0) = 0;
  tensor_expected_write(0, 1) = 1;
  tensor_expected_write(0, 2) = 2;
  // 2nd block
  tensor_expected_write(1, 0) = 40;
  tensor_expected_write(1, 1) = 50;
  tensor_expected_write(1, 2) = 60;

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.BeginMark();
    buffer.FinishMark();
    buffer.BeginMark();
      buffer.Add(40); buffer.Add(50); buffer.Add(60);
    buffer.FinishMark();
  buffer.FinishMark();

  // Get the resolved shape
  TensorShape resolved_shape;
  EXPECT_EQ(buffer.ResolveDenseShape(&resolved_shape, partial_shape, defaults_shape), Status::OK());
  EXPECT_EQ(resolved_shape, expected_shape);

  // Make the tensor from buffer
  Tensor tensor_actual(DT_INT32, resolved_shape);
  EXPECT_EQ(buffer.MakeDense(&tensor_actual, resolved_shape, defaults), Status::OK());
  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}


// Test with partial tensor shape, where the value buffer provides
// the complete shape information
TEST(ValueBufferTest, DenseShapeFromBuffer) {

  // Define shapes
  const TensorShape defaults_shape({1});
  const TensorShape expected_shape({2, 3});
  const PartialTensorShape partial_shape({-1, -1});

  // Define the default tensor
  Tensor defaults(DT_INT32, defaults_shape);
  auto defaults_flat = defaults.flat<int>();
  defaults_flat(0) = 100;

  // to allocate the proper amount of memory
  Tensor tensor_expected(DT_INT32, expected_shape);
  auto expected_flat = tensor_expected.flat<int>();
  for (int i_value = 0; i_value < expected_shape.num_elements(); ++i_value) {
    expected_flat(i_value) = i_value + 1;
  }

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.BeginMark();
      buffer.Add(1); buffer.Add(2); buffer.Add(3);
    buffer.FinishMark();
    buffer.BeginMark();
      buffer.Add(4); buffer.Add(5); buffer.Add(6);
    buffer.FinishMark();
  buffer.FinishMark();

  // Get the resolved shape
  TensorShape resolved_shape;
  EXPECT_EQ(buffer.ResolveDenseShape(&resolved_shape, partial_shape, defaults_shape), Status::OK());
  EXPECT_EQ(resolved_shape, expected_shape);

  // Make the tensor from buffer
  Tensor tensor_actual(DT_INT32, resolved_shape);
  EXPECT_EQ(buffer.MakeDense(&tensor_actual, resolved_shape, defaults), Status::OK());
  test::ExpectTensorEqual<int>(tensor_actual, tensor_expected);
}

// Test with string content -- since we use move instead of copy
TEST(ValueBufferTest, DenseStringContent) {

  // Define shapes
  const TensorShape defaults_shape({1});
  const TensorShape expected_shape({3});
  const TensorShape partial_shape(expected_shape);

  // Define defaults
  Tensor defaults(DT_STRING, defaults_shape);
  auto defaults_flat = defaults.flat<string>();
  defaults_flat(0) = "abc";

  // Define expected tensor
  Tensor tensor_expected(DT_STRING, expected_shape);
  auto expected_flat = tensor_expected.flat<string>();
  expected_flat(0) = "a";
  expected_flat(1) = "b";
  expected_flat(2) = "abc";

  // Initialize the buffer
  StringValueBuffer buffer;
  buffer.BeginMark();
    buffer.AddByRef("a"); buffer.AddByRef("b");
  buffer.FinishMark();

  // Get the resolved shape
  TensorShape resolved_shape;
  EXPECT_EQ(buffer.ResolveDenseShape(&resolved_shape, partial_shape, defaults_shape), Status::OK());
  EXPECT_EQ(resolved_shape, expected_shape);

  Tensor tensor_actual(DT_STRING, resolved_shape);
  EXPECT_EQ(buffer.MakeDense(&tensor_actual, resolved_shape, defaults), Status::OK());
  test::ExpectTensorEqual<string>(tensor_actual, tensor_expected);
}


// ------------------------------------------------------------
// Test Value buffer -- Sparse
// ------------------------------------------------------------
// Note, sparse does not make much sense in this case, but we support it for
// completeness
TEST(ValueBufferTest, Sparse1D) {

  // Define shapes
  const TensorShape expected_values_shape({2});
  const TensorShape expected_indices_shape({2});

  // Define expected values
  Tensor values_expected(DT_INT32, expected_values_shape);
  auto values_expected_flat = values_expected.flat<int>();
  values_expected_flat(0) = 1;
  values_expected_flat(1) = 4;

  // Define expected indices
  Tensor indices_expected(DT_INT64, expected_indices_shape);
  auto indices_expected_flat = indices_expected.flat<int64>();
  indices_expected_flat(0) = 0;
  indices_expected_flat(1) = 1;

  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.Add(1); buffer.Add(4);
  buffer.FinishMark();

  // Resolve shapes
  TensorShape actual_values_shape;
  TensorShape actual_indices_shape;
  EXPECT_EQ(buffer.GetSparseValueShape(&actual_values_shape), Status::OK());
  EXPECT_EQ(buffer.GetSparseIndexShape(&actual_indices_shape), Status::OK());

  // Check that shapes match
  EXPECT_EQ(actual_values_shape, expected_values_shape);
  EXPECT_EQ(actual_indices_shape, expected_indices_shape);

  // Retrieve tensors for indices and values
  Tensor values_actual(DT_INT32, actual_values_shape);
  Tensor indices_actual(DT_INT64, actual_indices_shape);
  EXPECT_EQ(buffer.MakeSparse(&values_actual, &indices_actual), Status::OK());

  // Check that values and indices match
  test::ExpectTensorEqual<int>(values_actual, values_expected);
  test::ExpectTensorEqual<int64>(indices_actual, indices_expected);
}

TEST(ValueBufferTest, Sparse2D) {

  // Define shapes
  const TensorShape expected_values_shape({4});
  const TensorShape expected_indices_shape({4, 2});

  // Define expected values
  Tensor values_expected(DT_INT32, expected_values_shape);
  auto values_expected_flat = values_expected.flat<int>();
  values_expected_flat(0) = 1;
  values_expected_flat(1) = 4;
  values_expected_flat(2) = 5;
  values_expected_flat(3) = 7;

  // Define expected indices
  Tensor indices_expected(DT_INT64, expected_indices_shape);
  auto indices_expected_tensor = indices_expected.tensor<int64, 2>();
  indices_expected_tensor(0, 0) = 0; indices_expected_tensor(0, 1) = 0;
  indices_expected_tensor(1, 0) = 1; indices_expected_tensor(1, 1) = 0;
  indices_expected_tensor(2, 0) = 1; indices_expected_tensor(2, 1) = 1;
  indices_expected_tensor(3, 0) = 1; indices_expected_tensor(3, 1) = 2;


  // Initialize the buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.BeginMark();
      buffer.Add(1);
    buffer.FinishMark();
    buffer.BeginMark();
      buffer.Add(4); buffer.Add(5); buffer.Add(7);
    buffer.FinishMark();
  buffer.FinishMark();

  // Resolve shapes
  TensorShape actual_values_shape;
  TensorShape actual_indices_shape;
  EXPECT_EQ(buffer.GetSparseValueShape(&actual_values_shape), Status::OK());
  EXPECT_EQ(buffer.GetSparseIndexShape(&actual_indices_shape), Status::OK());

  // Check that shapes match
  EXPECT_EQ(actual_values_shape, expected_values_shape);
  EXPECT_EQ(actual_indices_shape, expected_indices_shape);

  // Retrieve tensors for indices and values
  Tensor values_actual(DT_INT32, actual_values_shape);
  Tensor indices_actual(DT_INT64, actual_indices_shape);
  EXPECT_EQ(buffer.MakeSparse(&values_actual, &indices_actual), Status::OK());

  // Check that values and indices match
  test::ExpectTensorEqual<int>(values_actual, values_expected);
  test::ExpectTensorEqual<int64>(indices_actual, indices_expected);
}

// Test multiple nestings for sparse tensors
TEST(ValueBufferTest, Sparse4D) {

  // Define shapes
  const TensorShape expected_values_shape({7});
  const TensorShape expected_indices_shape({7, 4});

  // Define expected values
  Tensor values_expected(DT_INT32, expected_values_shape);
  auto values_expected_flat = values_expected.flat<int>();
  values_expected_flat(0) = 1;
  values_expected_flat(1) = 2;
  values_expected_flat(2) = 5;
  values_expected_flat(3) = 4;
  values_expected_flat(4) = 5;
  values_expected_flat(5) = 7;
  values_expected_flat(6) = 8;

  // Define expected indices
  Tensor indices_expected(DT_INT64, expected_indices_shape);
  auto indices_expected_tensor = indices_expected.tensor<int64, 2>();
  // index for 1st value
  indices_expected_tensor(0, 0) = 0;
  indices_expected_tensor(0, 1) = 0;
  indices_expected_tensor(0, 2) = 0;
  indices_expected_tensor(0, 3) = 0;
  // index for 2nd value
  indices_expected_tensor(1, 0) = 0;
  indices_expected_tensor(1, 1) = 0;
  indices_expected_tensor(1, 2) = 0;
  indices_expected_tensor(1, 3) = 1;
  // index for 3rd value
  indices_expected_tensor(2, 0) = 0;
  indices_expected_tensor(2, 1) = 0;
  indices_expected_tensor(2, 2) = 1;
  indices_expected_tensor(2, 3) = 0;
  // index for 4th value
  indices_expected_tensor(3, 0) = 1;
  indices_expected_tensor(3, 1) = 0;
  indices_expected_tensor(3, 2) = 0;
  indices_expected_tensor(3, 3) = 0;
  // index for 5th value
  indices_expected_tensor(4, 0) = 1;
  indices_expected_tensor(4, 1) = 0;
  indices_expected_tensor(4, 2) = 0;
  indices_expected_tensor(4, 3) = 1;
  // index for 6th value
  indices_expected_tensor(5, 0) = 1;
  indices_expected_tensor(5, 1) = 0;
  indices_expected_tensor(5, 2) = 0;
  indices_expected_tensor(5, 3) = 2;
  // index for 7th value
  indices_expected_tensor(6, 0) = 1;
  indices_expected_tensor(6, 1) = 1;
  indices_expected_tensor(6, 2) = 0;
  indices_expected_tensor(6, 3) = 0;

  // Define the data buffer
  IntValueBuffer buffer;
  buffer.BeginMark();
    buffer.BeginMark();
      buffer.BeginMark();
        buffer.BeginMark();
          buffer.Add(1); buffer.Add(2);
        buffer.FinishMark();
        buffer.BeginMark();
          buffer.Add(5);
        buffer.FinishMark();
      buffer.FinishMark();
    buffer.FinishMark();
    buffer.BeginMark();
      buffer.BeginMark();
        buffer.BeginMark();
          buffer.Add(4); buffer.Add(5); buffer.Add(7);
        buffer.FinishMark();
      buffer.FinishMark();
      buffer.BeginMark();
        buffer.BeginMark();
          buffer.Add(8);
        buffer.FinishMark();
      buffer.FinishMark();
    buffer.FinishMark();
  buffer.FinishMark();

  // Resolve shapes
  TensorShape actual_values_shape;
  TensorShape actual_indices_shape;
  EXPECT_EQ(buffer.GetSparseValueShape(&actual_values_shape), Status::OK());
  EXPECT_EQ(buffer.GetSparseIndexShape(&actual_indices_shape), Status::OK());

  // Check that shapes match
  EXPECT_EQ(actual_values_shape, expected_values_shape);
  EXPECT_EQ(actual_indices_shape, expected_indices_shape);

  // Retrieve tensors for indices and values
  Tensor values_actual(DT_INT32, actual_values_shape);
  Tensor indices_actual(DT_INT64, actual_indices_shape);
  EXPECT_EQ(buffer.MakeSparse(&values_actual, &indices_actual), Status::OK());
  LOG(INFO) << "Indices actual:   " << indices_actual.SummarizeValue(28);
  LOG(INFO) << "Indices expected: " << indices_expected.SummarizeValue(28);

  // Check that values and indices match
  test::ExpectTensorEqual<int>(values_actual, values_expected);
  test::ExpectTensorEqual<int64>(indices_actual, indices_expected);
}

// TODO: Test cases where the nesting level of the value buffer does not match, e.g.
//  buffer.BeginMark();
//    buffer.BeginMark();
//      buffer.Add(1);
//    buffer.FinishMark();
//    buffer.Add(4); buffer.Add(5); buffer.Add(7);
//  buffer.FinishMark();
// This should fail! probably in the add

// ------------------------------------------------------------
// Test Value buffer --
// ------------------------------------------------------------
TEST(ValueBufferTest, LatestValuesDoMatch) {
  IntValueBuffer buffer;
  buffer.Add(1);
  EXPECT_TRUE(buffer.ValuesMatchAtReverseIndex(buffer, 1));
}

TEST(ValueBufferTest, LatestValuesDoNotMatchValue) {
  IntValueBuffer this_buffer;
  this_buffer.Add(1);
  IntValueBuffer that_buffer;
  that_buffer.Add(2);
  EXPECT_TRUE(!this_buffer.ValuesMatchAtReverseIndex(that_buffer, 1));
}

TEST(ValueBufferTest, LatestValuesDoNotMatchType) {
  IntValueBuffer this_buffer;
  this_buffer.Add(1);
  FloatValueBuffer that_buffer;
  that_buffer.Add(2.0f);
  EXPECT_TRUE(!this_buffer.ValuesMatchAtReverseIndex(that_buffer, 1));
}

TEST(ValueBufferTest, LatestValueMatches) {
  StringValueBuffer buffer;
  buffer.Add("abc");
  EXPECT_TRUE(buffer.ValueMatchesAtReverseIndex("abc", 1));
  EXPECT_TRUE(!buffer.ValueMatchesAtReverseIndex("abcd", 1));
}

TEST(ValueBufferTest, LatestValueDoesNotMatchType) {
  IntValueBuffer buffer;
  buffer.Add(1);
  EXPECT_TRUE(!buffer.ValueMatchesAtReverseIndex("abc", 1));
}

}
}