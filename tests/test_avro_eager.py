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

if __name__ == "__main__":
  test.main()
