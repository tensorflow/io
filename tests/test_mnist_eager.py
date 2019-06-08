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
"""Tests for MNIST Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io.mnist as mnist_io # pylint: disable=wrong-import-position

def test_mnist_dataset():
  """Test case for MNIST Dataset.
  """
  mnist_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_mnist",
      "mnist.npz")
  with np.load(mnist_filename) as f:
    (x_test, y_test) = f['x_test'], f['y_test']

  image_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_mnist",
      "t10k-images-idx3-ubyte.gz")
  label_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_mnist",
      "t10k-labels-idx1-ubyte.gz")

  image_dataset = mnist_io.MNISTImageDataset(image_filename)
  label_dataset = mnist_io.MNISTLabelDataset(label_filename)

  i = 0
  for m_x in image_dataset:
    v_x = x_test[i]
    assert np.alltrue(v_x == m_x.numpy())
    i += 1
  assert i == len(y_test)

  i = 0
  for m_y in label_dataset:
    v_y = y_test[i]
    assert np.alltrue(v_y == m_y.numpy())
    i += 1
  assert i == len(y_test)

  dataset = mnist_io.MNISTDataset(
      image_filename, label_filename)

  i = 0
  for (m_x, m_y) in dataset:
    v_x = x_test[i]
    v_y = y_test[i]
    assert np.alltrue(v_y == m_y.numpy())
    assert np.alltrue(v_x == m_x.numpy())
    i += 1
  assert i == len(y_test)

if __name__ == "__main__":
  test.main()
