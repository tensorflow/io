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
"""Tests for HDF5Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow import dtypes  # pylint: disable=wrong-import-position
from tensorflow import errors  # pylint: disable=wrong-import-position
from tensorflow import test    # pylint: disable=wrong-import-position
from tensorflow.compat.v1 import data # pylint: disable=wrong-import-position

import tensorflow_io.hdf5 as hdf5_io # pylint: disable=wrong-import-position

class HDF5DatasetTest(test.TestCase):
  """HDF5DatasetTest"""

  def test_hdf5_invalid_dataset(self):
    """test_hdf5_invalid_dataset"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_hdf5", "tdset.h5")
    filename = "file://" + filename
    dataset = hdf5_io.HDF5Dataset(
        filename,
        '/invalid',
        dtype=dtypes.int32,
        shape=tf.TensorShape([1, 20]),
        start=0,
        stop=10)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError, "unable to open dataset"):
        sess.run(get_next)

  def test_hdf5_dataset_int32(self):
    """Test case for HDF5Dataset."""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_hdf5", "tdset.h5")
    filename = "file://" + filename
    column = '/dset1'
    dtype = dtypes.int32
    shape = tf.TensorShape([None, 20])

    dataset = hdf5_io.HDF5Dataset(
        filename, column, start=0, stop=10, dtype=dtype, shape=shape).apply(
            tf.data.experimental.unbatch())
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(10):
        v0 = list([np.asarray([v for v in range(i, i + 20)])])
        vv = sess.run(get_next)
        self.assertAllEqual(v0, [vv])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


  def test_hdf5_dataset_int32_zlib(self):
    """Test case for HDF5Dataset with zlib."""
    # Note the file is generated with tdset.h5:
    # with h5py.File('compressed_h5.h5', 'w') as output_f:
    #   output_f.create_dataset(
    #       '/dset1', data=h5f['/dset1'][()], compression='gzip')
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_hdf5", "compressed_h5.h5")
    filename = "file://" + filename
    column = '/dset1'
    dtype = dtypes.int32
    shape = tf.TensorShape([None, 20])

    dataset = hdf5_io.HDF5Dataset(
        filename, column, start=0, stop=10, dtype=dtype, shape=shape).apply(
            tf.data.experimental.unbatch())
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(10):
        v0 = list([np.asarray([v for v in range(i, i + 20)])])
        vv = sess.run(get_next)
        self.assertAllEqual(v0, [vv])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


  def test_hdf5_dataset(self):
    """Test case for HDF5Dataset."""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_hdf5", "tdset.h5")
    filename = "file://" + filename
    column = '/dset2'
    dtype = dtypes.float64
    shape = tf.TensorShape([None, 20])

    dataset = hdf5_io.HDF5Dataset(
        filename, column, start=0, stop=30, dtype=dtype, shape=shape).apply(
            tf.data.experimental.unbatch())
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(30):
        v0 = np.asarray([[i + 1e-04 * v for v in range(20)]],
                        dtype=np.float64)
        vv = sess.run(get_next)
        self.assertAllEqual(v0, [vv])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

if __name__ == "__main__":
  test.main()
