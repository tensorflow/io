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

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow import dtypes  # pylint: disable=wrong-import-position
from tensorflow import errors  # pylint: disable=wrong-import-position
from tensorflow import test    # pylint: disable=wrong-import-position
from tensorflow.compat.v1 import data # pylint: disable=wrong-import-position

import tensorflow_io.parquet as parquet_io # pylint: disable=wrong-import-position

class ParquetDatasetTest(test.TestCase):
  """ParquetDatasetTest"""
  def test_parquet_dataset(self):
    """Test case for ParquetDataset.

    Note: The sample file is generated from:
    `parquet-cpp/examples/low-level-api/reader_writer`
    This test extracts columns of [0, 1, 2, 4, 5]
    with column data types of [bool, int32, int64, float, double].
    Please check `parquet-cpp/examples/low-level-api/reader-writer.cc`
    to find details of how records are generated:
    Column 0 (bool): True for even rows and False otherwise.
    Column 1 (int32): Equal to row_index.
    Column 2 (int64): Equal to row_index * 1000 * 1000 * 1000 * 1000.
    Column 4 (float): Equal to row_index * 1.1.
    Column 5 (double): Equal to row_index * 1.1111111.
    """
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
    output_types = (
        dtypes.bool, dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64)
    num_repeats = 2

    dataset = parquet_io.ParquetDataset(
        [filename], columns, output_types).repeat(num_repeats)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):  # Dataset is repeated.
        for i in range(500): # 500 rows.
          v0 = ((i % 2) == 0)
          v1 = i
          v2 = i * 1000 * 1000 * 1000 * 1000
          v4 = 1.1 * i
          v5 = 1.1111111 * i
          vv = sess.run(get_next)
          self.assertAllClose((v0, v1, v2, v4, v5), vv)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

    dataset = parquet_io.ParquetDataset(
        [filename], columns, output_types, batch=1)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(500):
        v0 = ((i % 2) == 0)
        v1 = i
        v2 = i * 1000 * 1000 * 1000 * 1000
        v4 = 1.1 * i
        v5 = 1.1111111 * i
        vv = sess.run(get_next)
        self.assertAllClose(([v0], [v1], [v2], [v4], [v5]), vv)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

    dataset = parquet_io.ParquetDataset(
        [filename, filename], columns, output_types, batch=3)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for ii in range(0, 999, 3):
        v0, v1, v2, v4, v5 = [], [], [], [], []
        for i in [ii % 500, (ii + 1) % 500, (ii + 2) % 500]:
          v0.append((i % 2) == 0)
          v1.append(i)
          v2.append(i * 1000 * 1000 * 1000 * 1000)
          v4.append(1.1 * i)
          v5.append(1.1111111 * i)
        vv = sess.run(get_next)
        self.assertAllClose((v0, v1, v2, v4, v5), vv)
      i = 999 % 500
      v0 = ((i % 2) == 0)
      v1 = i
      v2 = i * 1000 * 1000 * 1000 * 1000
      v4 = 1.1 * i
      v5 = 1.1111111 * i
      vv = sess.run(get_next)
      self.assertAllClose(([v0], [v1], [v2], [v4], [v5]), vv)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

    # With compression
    filename = filename + '.gz'
    dataset = parquet_io.ParquetDataset(
        [filename], columns, output_types).repeat(num_repeats)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):  # Dataset is repeated.
        for i in range(500): # 500 rows.
          v0 = ((i % 2) == 0)
          v1 = i
          v2 = i * 1000 * 1000 * 1000 * 1000
          v4 = 1.1 * i
          v5 = 1.1111111 * i
          vv = sess.run(get_next)
          self.assertAllClose((v0, v1, v2, v4, v5), vv)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
if __name__ == "__main__":
  test.main()
