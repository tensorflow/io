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
"""Tests for read_parquet and ParquetDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import numpy as np

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io as tfio # pylint: disable=wrong-import-position

filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_parquet",
    "parquet_cpp_example.parquet")
filename = "file://" + filename

# Note: The sample file is generated from the following after apply patch
# tests/test_parquet/parquet_cpp_example.patch:
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
  """Test case for read_parquet.

  """
  parquet = tfio.IOTensor.from_parquet(filename)
  columns = [
      'boolean_field',
      'int32_field',
      'int64_field',
      'int96_field',
      'float_field',
      'double_field',
      'ba_field',
      'flba_field']
  assert parquet.columns == columns
  p0 = parquet('boolean_field')
  p1 = parquet('int32_field')
  p2 = parquet('int64_field')
  p4 = parquet('float_field')
  p5 = parquet('double_field')
  p6 = parquet('ba_field')
  p7 = parquet('flba_field')
  assert p0.dtype == tf.bool
  assert p1.dtype == tf.int32
  assert p2.dtype == tf.int64
  assert p4.dtype == tf.float32
  assert p5.dtype == tf.float64
  assert p6.dtype == tf.string
  assert p7.dtype == tf.string

  for i in range(500): # 500 rows.
    v0 = ((i % 2) == 0)
    v1 = i
    v2 = i * 1000 * 1000 * 1000 * 1000
    v4 = 1.1 * i
    v5 = 1.1111111 * i
    v6 = b"parquet%03d" % i
    v7 = bytearray(b"").join([bytearray((i % 256,)) for _ in range(10)])
    assert v0 == p0[i].numpy()
    assert v1 == p1[i].numpy()
    assert v2 == p2[i].numpy()
    assert np.isclose(v4, p4[i].numpy())
    assert np.isclose(v5, p5[i].numpy())
    assert v6 == p6[i].numpy()
    assert v7 == p7[i].numpy()

  # test parquet dataset
  columns = [
      'boolean_field',
      'int32_field',
      'int64_field',
      'float_field',
      'double_field',
      'ba_field',
      'flba_field']
  dataset = tfio.IODataset.from_parquet(filename, columns)
  i = 0
  for v in dataset:
    v0 = ((i % 2) == 0)
    v1 = i
    v2 = i * 1000 * 1000 * 1000 * 1000
    v4 = 1.1 * i
    v5 = 1.1111111 * i
    v6 = b"parquet%03d" % i
    v7 = bytearray(b"").join([bytearray((i % 256,)) for _ in range(10)])
    p0 = v['boolean_field']
    p1 = v['int32_field']
    p2 = v['int64_field']
    p4 = v['float_field']
    p5 = v['double_field']
    p6 = v['ba_field']
    p7 = v['flba_field']
    assert v0 == p0.numpy()
    assert v1 == p1.numpy()
    assert v2 == p2.numpy()
    assert np.isclose(v4, p4.numpy())
    assert np.isclose(v5, p5.numpy())
    assert v6 == p6.numpy()
    assert v7 == p7.numpy()
    i += 1

def test_parquet_partition():
  """test_parquet_partition"""
  for capacity in [
      1, 2, 3,
      11, 12, 13,
      50, 51, 100, 200]:
    parquet = tfio.IOTensor.from_parquet(
        filename, capacity=capacity)
    assert np.all(
        parquet("int32_field").to_tensor().numpy() == list(range(500)))
    for step in [
        1, 2, 3,
        10, 11, 12, 13,
        50, 51, 52, 53]:
      indices = list(range(0, 100, step))
      for start, stop in zip(indices, indices[1:] + [100]):
        expected = list(range(start, stop))
        items = parquet("int32_field")[start:stop]
        assert np.all(items.numpy() == expected)

def test_parquet_dataset_ordered_dict():
  """Test case for order and dict of parquet dataset"""
  parquet = tfio.IODataset.from_parquet(filename)
  assert parquet.element_spec == collections.OrderedDict([
      (b'boolean_field', tf.TensorSpec(shape=(), dtype=tf.bool)),
      (b'int32_field', tf.TensorSpec(shape=(), dtype=tf.int32)),
      (b'int64_field', tf.TensorSpec(shape=(), dtype=tf.int64)),
      (b'int96_field', tf.TensorSpec(shape=(), dtype=tf.int64)),
      (b'float_field', tf.TensorSpec(shape=(), dtype=tf.float32)),
      (b'double_field', tf.TensorSpec(shape=(), dtype=tf.float64)),
      (b'ba_field', tf.TensorSpec(shape=(), dtype=tf.string)),
      (b'flba_field', tf.TensorSpec(shape=(), dtype=tf.string))])

if __name__ == "__main__":
  test.main()
