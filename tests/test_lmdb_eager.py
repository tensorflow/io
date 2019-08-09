#  Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for LMDBDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io.lmdb as lmdb_io # pylint: disable=wrong-import-position

def test_lmdb_read_from_file():
  """test_read_from_file"""
  # Copy database out because we need the path to be writable to use locks.
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_lmdb", "data.mdb")

  for capacity in [1, 2, 3, 4, 5, 10, 100, 1000]:
    tmp_path = tempfile.mkdtemp()
    filename = os.path.join(tmp_path, "data.mdb")
    shutil.copy(path, filename)
    lmdb_dataset = lmdb_io.LMDBDataset(
        filename, capacity=capacity).apply(
            tf.data.experimental.unbatch())
    ii = 0
    for vv in lmdb_dataset:
      i = ii % 10
      k, v = vv
      assert k.numpy() == str(i).encode()
      assert v.numpy() == str(chr(ord("a") + i)).encode()
      ii += 1
    shutil.rmtree(tmp_path)

  tmp_path = tempfile.mkdtemp()
  filename = os.path.join(tmp_path, "data.mdb")
  shutil.copy(path, filename)

  key, val = lmdb_io.read_lmdb(filename)
  assert key.shape == [10]
  assert val.shape == [10]
  for i in range(10):
    assert key[i].numpy() == str(i).encode()
    assert val[i].numpy() == str(chr(ord("a") + i)).encode()

  shutil.rmtree(tmp_path)


if __name__ == "__main__":
  test.main()
