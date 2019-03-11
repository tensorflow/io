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

import tensorflow
tensorflow.compat.v1.disable_eager_execution()

from tensorflow import errors         # pylint: disable=wrong-import-position
from tensorflow import test           # pylint: disable=wrong-import-position
from tensorflow.compat.v1 import data # pylint: disable=wrong-import-position

from tensorflow_io.mnist.python.ops import mnist_dataset_ops # pylint: disable=wrong-import-position


class MNISTDatasetTest(test.TestCase):
  """MNISTDatasetTest"""
  def test_mnist_dataset(self):
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

    image_dataset = mnist_dataset_ops.MNISTImageDataset(
        [image_filename], compression_type="GZIP")
    label_dataset = mnist_dataset_ops.MNISTLabelDataset(
        [label_filename], compression_type="GZIP")

    dataset = mnist_dataset_ops.MNISTDataset(
        image_filename, label_filename, compression_type="GZIP")

    self.assertEqual(image_dataset.output_shapes.as_list(), [None, None])
    self.assertEqual(label_dataset.output_shapes.as_list(), [])

    iterator = data.Dataset.zip(
        (image_dataset, label_dataset)).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      l = len(y_test)
      for i in range(l):
        v_x = x_test[i]
        v_y = y_test[i]
        m_x, m_y = sess.run(get_next)
        self.assertEqual(v_y, m_y)
        self.assertAllEqual(v_x, m_x)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      l = len(y_test)
      for i in range(l):
        v_x = x_test[i]
        v_y = y_test[i]
        m_x, m_y = sess.run(get_next)
        self.assertEqual(v_y, m_y)
        self.assertAllEqual(v_x, m_x)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

if __name__ == "__main__":
  test.main()
