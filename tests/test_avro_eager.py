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
"""Tests for AvroDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io.avro as avro_io # pylint: disable=wrong-import-position

def test_avro():
  """test_list_avro_columns."""
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

  specs = avro_io.list_avro_columns(filename, schema)
  assert specs["im"].dtype == tf.float64
  assert specs["re"].dtype == tf.float64

  v0 = avro_io.read_avro(filename, schema, specs["im"])
  v1 = avro_io.read_avro(filename, schema, specs["re"])
  for i in range(100):
    (im, re) = (i + 100, i * 100)
    assert v0[i].numpy() == im
    assert v1[i].numpy() == re

  for capacity in [10, 20, 50, 100, 1000, 2000]:
    dataset = tf.compat.v2.data.Dataset.zip(
        (
            avro_io.AvroDataset(filename, schema, "im", capacity=capacity),
            avro_io.AvroDataset(filename, schema, "re", capacity=capacity)
        )
    ).apply(tf.data.experimental.unbatch())
    i = 0
    for vv in dataset:
      v0, v1 = vv
      (im, re) = (i + 100, i * 100)
      assert v0.numpy() == im
      assert v1.numpy() == re
      i += 1

if __name__ == "__main__":
  test.main()
