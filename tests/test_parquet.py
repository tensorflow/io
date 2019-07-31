# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for ParquetDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pytest
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow_io.parquet as parquet_io # pylint: disable=wrong-import-position

# Note: The sample file is generated from:
# `parquet-cpp/examples/low-level-api/reader_writer`
# This test extracts columns of [0, 1, 2, 4, 5]
# with column data types of [bool, int32, int64, float, double].
# Please check `parquet-cpp/examples/low-level-api/reader-writer.cc`
# to find details of how records are generated:
# Column 0 (bool): True for even rows and False otherwise.
# Column 1 (int32): Equal to row_index.
# Column 2 (int64): Equal to row_index * 1000 * 1000 * 1000 * 1000.
# Column 4 (float): Equal to row_index * 1.1.
# Column 5 (double): Equal to row_index * 1.1111111.
def test_parquet():
  """Test case for ParquetDataset."""
  filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_parquet",
      "parquet_cpp_example.parquet")
  filename = "file://" + filename

  columns = [
      'boolean_field',
      'int32_field',
      'int64_field',
      'float_field',
      'double_field']
  dtypes = [
      tf.bool,
      tf.int32,
      tf.int64,
      tf.float32,
      tf.double]

  dataset = tf.compat.v2.data.Dataset.zip(
      tuple([parquet_io.ParquetDataset(
          filename, column, dtype=dtype,
          start=0, stop=500) for (
              column, dtype) in zip(columns, dtypes)])).apply(
                  tf.data.experimental.unbatch())

  iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
  init_op = iterator.initializer
  get_next = iterator.get_next()
  with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    for i in range(500):
      v0 = ((i % 2) == 0)
      v1 = i
      v2 = i * 1000 * 1000 * 1000 * 1000
      v4 = 1.1 * i
      v5 = 1.1111111 * i
      p0, p1, p2, p4, p5 = sess.run(get_next)
      assert v0 == p0
      assert v1 == p1
      assert v2 == p2
      assert np.isclose(v4, p4)
      assert np.isclose(v5, p5)
    with pytest.raises(tf.errors.OutOfRangeError):
      sess.run(get_next)

if __name__ == "__main__":
  test.main()
