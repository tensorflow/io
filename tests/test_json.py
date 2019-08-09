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
"""Tests for JSON Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow_io.json as json_io # pylint: disable=wrong-import-position


def test_json_dataset():
  """Test case for JSONDataset."""
  filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_json",
      "feature.json")
  filename = "file://" + filename

  num_repeats = 2
  dataset = tf.compat.v2.data.Dataset.zip(
      (
          json_io.JSONDataset(filename, "floatfeature", dtype=tf.float64),
          json_io.JSONDataset(filename, "integerfeature", dtype=tf.int64)
      )).repeat(num_repeats).apply(tf.data.experimental.unbatch())

  iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
  init_op = iterator.initializer
  get_next = iterator.get_next()

  test_json = [(1.1, 2), (2.1, 3)]
  with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    for _ in range(num_repeats):
      for i in range(2):
        (floatf, intf) = test_json[i]
        vv = sess.run(get_next)
        np.allclose((floatf, intf), vv)
    with pytest.raises(tf.errors.OutOfRangeError):
      sess.run(get_next)

if __name__ == "__main__":
  test.main()
