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
"""Tests for CIFARDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
from six.moves.urllib.request import urlopen

import tensorflow
tensorflow.compat.v1.disable_eager_execution()

from tensorflow import errors         # pylint: disable=wrong-import-position
from tensorflow import test           # pylint: disable=wrong-import-position
from tensorflow.compat.v1 import data # pylint: disable=wrong-import-position

import tensorflow_io.cifar as cifar_io # pylint: disable=wrong-import-position

class CIFARDatasetTest(test.TestCase):
  """CIFARDatasetTest"""

  def test_cifar_10_dataset(self):
    """Test case for CIFARDataset.
    """
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    filedata = urlopen(url)
    f, filename = tempfile.mkstemp()
    os.write(f, filedata.read())
    os.close(f)
    (x_train, y_train), (
        x_test, y_test) = tensorflow.keras.datasets.cifar10.load_data()

    num_repeats = 2

    dataset = cifar_io.CIFAR10Dataset(filename, batch=3).repeat(
        num_repeats)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):  # Dataset is repeated.
        for i in range(16666):
          image, label = sess.run(get_next)
          self.assertAllEqual(image[0], x_train[i*3+0])
          self.assertEqual(label[0], y_train[i*3+0])
          self.assertAllEqual(image[1], x_train[i*3+1])
          self.assertEqual(label[1], y_train[i*3+1])
          self.assertAllEqual(image[2], x_train[i*3+2])
          self.assertEqual(label[2], y_train[i*3+2])
        image, label = sess.run(get_next)
        self.assertAllEqual(image[0], x_train[49998])
        self.assertEqual(label[0], y_train[49998])
        self.assertAllEqual(image[1], x_train[49999])
        self.assertEqual(label[1], y_train[49999])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

    dataset = cifar_io.CIFAR10Dataset(filename, test=True).repeat(
        num_repeats)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):  # Dataset is repeated.
        for i in range(10000):
          image, label = sess.run(get_next)
          self.assertAllEqual(image, x_test[i])
          self.assertEqual(label, y_test[i])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # Skip the test for now as it takes time.
  def _test_cifar_100_dataset(self):
    """Test case for CIFARDataset.
    """
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
    filedata = urlopen(url)
    f, filename = tempfile.mkstemp()
    os.write(f, filedata.read())
    os.close(f)
    (x_train, y_train), (
        x_test, y_test) = tensorflow.keras.datasets.cifar100.load_data(
            label_mode='coarse')

    num_repeats = 2

    dataset = cifar_io.CIFAR100Dataset(filename, mode="coarse").repeat(
        num_repeats)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):  # Dataset is repeated.
        for i in range(50000):
          image, label = sess.run(get_next)
          self.assertAllEqual(image, x_train[i])
          self.assertEqual(label, y_train[i])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

    dataset = cifar_io.CIFAR100Dataset(
        filename, test=True, mode="coarse").repeat(
            num_repeats)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):  # Dataset is repeated.
        for i in range(10000):
          image, label = sess.run(get_next)
          self.assertAllEqual(image, x_test[i])
          self.assertEqual(label, y_test[i])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

if __name__ == "__main__":
  test.main()
