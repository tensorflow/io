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
import shutil
import tempfile
import numpy as np
import h5py

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io as tfio # pylint: disable=wrong-import-position
from tensorflow_io import hdf5 as hdf5_io # pylint: disable=wrong-import-position

def test_hdf5_list_dataset():
  """test_hdf5_list_dataset"""
  filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_hdf5", "h5ex_g_traverse.h5")

  # Without file:// file will be opened directly, otherwise
  # file will be opened in memory.
  for filename in [filename, "file://" + filename]:
    hdf5 = tfio.IOTensor.from_hdf5(filename)
    assert hdf5('/group1/dset1').dtype == tf.int32
    assert hdf5('/group1/dset1').shape == [1, 1]
    assert hdf5('/group1/group3/dset2').dtype == tf.int32
    assert hdf5('/group1/group3/dset2').shape == [1, 1]

def test_hdf5_read_dataset():
  """test_hdf5_list_dataset"""
  filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_hdf5", "tdset.h5")

  for filename in [filename, "file://" + filename]:
    hdf5 = tfio.IOTensor.from_hdf5(filename)
    assert hdf5('/dset1').dtype == tf.int32
    assert hdf5('/dset1').shape == [10, 20]
    assert hdf5('/dset2').dtype == tf.float64
    assert hdf5('/dset2').shape == [30, 20]

    p1 = hdf5('/dset1')
    for i in range(10):
      vv = list([np.asarray([v for v in range(i, i + 20)])])
      assert np.all(p1[i].numpy() == vv)

  dataset = tfio.IOTensor.from_hdf5(filename)('/dset1').to_dataset()
  i = 0
  for p in dataset:
    vv = list([np.asarray([v for v in range(i, i + 20)])])
    assert np.all(p.numpy() == vv)
    i += 1

def test_hdf5_u1():
  """test_hdf5_u1"""
  tmp_path = tempfile.mkdtemp()
  filename = os.path.join(tmp_path, "test.h5")

  data = np.array([1, 1, 1, 1])

  with h5py.File(filename, 'w') as f:
    f.create_dataset('uint', data=data, dtype='u1')
    f.create_dataset('int', data=data, dtype='i4')
    f.create_dataset('float', data=data)

  hdf5 = tfio.IOTensor.from_hdf5(filename)
  np.all(hdf5('/uint').to_tensor().numpy() == data)

  dataset = hdf5_io.list_hdf5_datasets(filename)
  keys = list(dataset.keys())
  keys.sort()
  np.all(keys == ['float', 'int', 'uint'])

  shutil.rmtree(tmp_path)


if __name__ == "__main__":
  test.main()
