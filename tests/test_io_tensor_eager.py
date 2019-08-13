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
"""Tests for IOTensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io.core.python.ops.io_tensor as io_tensor # pylint: disable=wrong-import-position

def test_mnist_io_tensor():
  """Test case for MNIST."""
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

  images = io_tensor.IOTensor.from_mnist(image_filename)
  labels = io_tensor.IOTensor.from_mnist(label_filename)

  for i, v in enumerate(x_test):
    assert np.alltrue(images[i].numpy() == v)
  assert len(images) == len(x_test)

  assert np.alltrue(images.to_tensor().numpy() == x_test)

  for i, v in enumerate(y_test):
    assert np.alltrue(labels[i].numpy() == v)
  assert len(labels) == len(y_test)

  assert np.alltrue(labels.to_tensor().numpy() == y_test)

if __name__ == "__main__":
  test.main()
