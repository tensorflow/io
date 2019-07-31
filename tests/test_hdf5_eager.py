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
"""Tests for HDF5."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io.hdf5 as hdf5_io # pylint: disable=wrong-import-position

def test_hdf5_list_dataset():
  """test_hdf5_list_dataset"""
  filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_hdf5", "h5ex_g_traverse.h5")

  # Without file:// file will be opened directly, otherwise
  # file will be opened in memory.
  for filename in [filename, "file://" + filename]:
    specs = hdf5_io.list_hdf5_datasets(filename)
    assert specs['/group1/dset1'].dtype == tf.int32
    assert specs['/group1/dset1'].shape == tf.TensorShape([1, 1])
    assert specs['/group1/group3/dset2'].dtype == tf.int32
    assert specs['/group1/group3/dset2'].shape == tf.TensorShape([1, 1])

def test_hdf5_read_dataset():
  """test_hdf5_list_dataset"""
  filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_hdf5", "tdset.h5")

  for filename in [filename, "file://" + filename]:
    specs = hdf5_io.list_hdf5_datasets(filename)
    assert specs['/dset1'].dtype == tf.int32
    assert specs['/dset1'].shape == tf.TensorShape([10, 20])
    assert specs['/dset2'].dtype == tf.float64
    assert specs['/dset2'].shape == tf.TensorShape([30, 20])

    p1 = hdf5_io.read_hdf5(filename, specs['/dset1'])
    assert p1.dtype == tf.int32
    assert p1.shape == tf.TensorShape([10, 20])
    for i in range(10):
      vv = list([np.asarray([v for v in range(i, i + 20)])])
      assert np.all(p1[i].numpy() == vv)

  dataset = hdf5_io.HDF5Dataset(filename, '/dset1').apply(
      tf.data.experimental.unbatch())
  i = 0
  for p in dataset:
    vv = list([np.asarray([v for v in range(i, i + 20)])])
    assert np.all(p.numpy() == vv)
    i += 1


if __name__ == "__main__":
  test.main()
