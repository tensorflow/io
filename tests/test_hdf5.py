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
        [filename], ['/invalid', '/invalid2'], (dtypes.int32, dtypes.int32))
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError, "unable to open dataset /invalid"):
        sess.run(get_next)

  def test_hdf5_dataset(self):
    """Test case for HDF5Dataset."""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_hdf5", "tdset.h5")
    filename = "file://" + filename
    # TODO: boilerplate, to be replaced
    columns = ['/dset1', '/dset2']
    output_types = (dtypes.int32, dtypes.float64)
    num_repeats = 2

    dataset = hdf5_io.HDF5Dataset(
        [filename], columns, output_types).repeat(num_repeats)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      with self.assertRaisesRegexp(
          errors.UnimplementedError, "HDF5 is currently not supported"):
        sess.run(get_next)

if __name__ == "__main__":
  test.main()
