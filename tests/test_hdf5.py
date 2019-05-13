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
import tensorflow
tensorflow.compat.v1.disable_eager_execution()

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
        [filename],
        ['/invalid', '/invalid2'],
        [dtypes.int32, dtypes.int32],
        [(1, 20), (1, 30)])
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError, "unable to open dataset /invalid"):
        sess.run(get_next)

  def test_hdf5_dataset_int32(self):
    """Test case for HDF5Dataset."""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_hdf5", "tdset.h5")
    filename = "file://" + filename
    columns = ['/dset1']
    output_types = [dtypes.int32]
    output_shapes = [(1, 20)]

    dataset = hdf5_io.HDF5Dataset(
        [filename], columns, output_types, output_shapes)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(10):
        v0 = list([np.asarray([v for v in range(i, i + 20)])])
        vv = sess.run(get_next)
        self.assertAllEqual(v0, vv)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


  def test_hdf5_dataset(self):
    """Test case for HDF5Dataset."""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_hdf5", "tdset.h5")
    filename = "file://" + filename
    columns = ['/dset2']
    output_types = [dtypes.float32]
    output_shapes = [(1, 20)]

    dataset = hdf5_io.HDF5Dataset(
        [filename], columns, output_types, output_shapes, batch=1)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(30):
        v0 = list(
            [np.asarray([[i + 1e-04 * v for v in range(20)]],
                        dtype=np.float32)])
        vv = sess.run(get_next)
        self.assertAllEqual(v0, vv)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

if __name__ == "__main__":
  test.main()
