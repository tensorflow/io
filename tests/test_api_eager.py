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
"""High level API tests for tensorflow_io.Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow.compat.v2.data as data # pylint: disable=wrong-import-position
import tensorflow_io as tfio # pylint: disable=wrong-import-position

def test_dataset():
  """test_dataset"""
  def assert_equal(io_v, tf_v):
    assert isinstance(io_v, tfio.Dataset)
    assert np.all(tf.stack(
        [v for v in io_v]).numpy() == tf.stack(
            [v for v in tf_v]).numpy())

  io_v = tfio.Dataset.range(1, 4)
  tf_v = data.Dataset.range(1, 4)
  assert_equal(io_v, tf_v)

  io_v = io_v.concatenate(tfio.Dataset.range(4, 8))
  tf_v = tf_v.concatenate(data.Dataset.range(4, 8))
  assert_equal(io_v, tf_v)

  io_v = io_v.filter(lambda x: x < 3)
  tf_v = tf_v.filter(lambda x: x < 3)
  assert_equal(io_v, tf_v)

  io_v = io_v.batch(3).rebatch(5).rebatch(3)
  tf_v = tf_v.batch(3)
  assert_equal(io_v, tf_v)


if __name__ == "__main__":
  test.main()
