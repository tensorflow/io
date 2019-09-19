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
"""Tests for tfio.IOTensor.from_avro."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io as tfio # pylint: disable=wrong-import-position

def test_avro():
  """test_avro"""
  # The test.bin was created from avro/lang/c++/examples/datafile.cc.
  filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_avro", "test.bin")
  filename = "file://" + filename

  schema_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_avro", "cpx.json")
  with open(schema_filename, 'r') as f:
    schema = f.read()

  avro = tfio.IOTensor.from_avro(filename, schema)
  assert avro("im").dtype == tf.float64
  assert avro("im").shape == [100]
  assert avro("re").dtype == tf.float64
  assert avro("re").shape == [100]

  assert np.all(
      avro("im").to_tensor().numpy() == [100.0 + i for i in range(100)])
  assert np.all(
      avro("re").to_tensor().numpy() == [100.0 * i for i in range(100)])

  dataset = avro.to_dataset()
  i = 0
  for v in dataset:
    re, im = v
    assert im.numpy() == 100.0 + i
    assert re.numpy() == 100.0 * i
    i += 1

def test_avro_partition():
  """test_avro_partition"""
  # The test.bin was created from avro/lang/c++/examples/datafile.cc.
  filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_avro", "test.bin")
  filename = "file://" + filename

  schema_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_avro", "cpx.json")
  with open(schema_filename, 'r') as f:
    schema = f.read()
  for capacity in [
      1, 2, 3,
      11, 12, 13,
      50, 51, 100]:
    avro = tfio.IOTensor.from_avro(
        filename, schema, capacity=capacity)
    assert np.all(
        avro("im").to_tensor().numpy() == [100.0 + i for i in range(100)])
    assert np.all(
        avro("re").to_tensor().numpy() == [100.0 * i for i in range(100)])
    for step in [
        1, 2, 3,
        10, 11, 12, 13,
        50, 51, 52, 53]:
      indices = list(range(0, 100, step))
      for start, stop in zip(indices, indices[1:] + [100]):
        im_expected = [100.0 + i for i in range(start, stop)]
        im_items = avro("im")[start:stop]
        assert np.all(im_items.numpy() == im_expected)

        re_expected = [100.0 * i for i in range(start, stop)]
        re_items = avro("re")[start:stop]
        assert np.all(re_items.numpy() == re_expected)

def test_avro_dataset_partition():
  """test_avro_dataset_partition"""
  # The test.bin was created from avro/lang/c++/examples/datafile.cc.
  filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_avro", "test.bin")
  filename = "file://" + filename

  schema_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_avro", "cpx.json")
  with open(schema_filename, 'r') as f:
    schema = f.read()
  for capacity in [1, 2, 3, 11, 12, 13, 50, 51, 100]:
    avro = tfio.IOTensor.from_avro(
        filename, schema, capacity=capacity)
    dataset = avro.to_dataset()
    i = 0
    for v in dataset:
      re, im = v
      assert im.numpy() == 100.0 + i
      assert re.numpy() == 100.0 * i
      i += 1

if __name__ == "__main__":
  test.main()
