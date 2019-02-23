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

from tensorflow.compat.v1 import data
from tensorflow import dtypes
from tensorflow import errors

from tensorflow_io.mnist.python.ops import mnist_dataset_ops

from tensorflow import test

class MNISTDatasetTest(test.TestCase):

  def test_mnist_dataset(self):
    """Test case for MNIST Dataset.
    """
    image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_mnist", "t10k-images-idx3-ubyte.gz")
    label_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_mnist", "t10k-labels-idx1-ubyte.gz")

    image_dataset = mnist_dataset_ops.MNISTImageDataset([image_filename], compression_type="GZIP")
    label_dataset = mnist_dataset_ops.MNISTLabelDataset([label_filename], compression_type="GZIP")
    iterator = data.Dataset.zip((image_dataset, label_dataset)).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    mnist_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_mnist", "mnist.npz")
    with np.load(mnist_filename) as f:
      (x_test, y_test) = f['x_test'], f['y_test']
    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(0, len(y_test)):
        v_x = x_test[i];
        v_y = y_test[i];
        m_x, m_y = sess.run(get_next)
        self.assertEqual(v_y, m_y)
        self.assertAllEqual(v_x, m_x)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
