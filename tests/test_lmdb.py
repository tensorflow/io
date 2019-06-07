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
"""Tests for LMDBDatasetOp."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import errors # pylint: disable=wrong-import-position
from tensorflow import test   # pylint: disable=wrong-import-position

import tensorflow_io.lmdb as lmdb_io # pylint: disable=wrong-import-position


class LMDBDatasetTest(test.TestCase):
  """LMDBDatasetTest"""

  def test_read_from_file(self):
    """test_read_from_file"""
    super(LMDBDatasetTest, self).setUp()
    # Copy database out because we need the path to be writable to use locks.
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_lmdb", "data.mdb")
    self.db_path = os.path.join(self.get_temp_dir(), "data.mdb")
    shutil.copy(path, self.db_path)

    filename = self.db_path

    num_repeats = 2

    dataset = lmdb_io.LMDBDataset([filename]).repeat(num_repeats)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):
        for i in range(10):
          k = str(i).encode()
          v = str(chr(ord("a") + i)).encode()
          self.assertEqual((k, v), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def test_read_from_file_with_batch(self):
    """test_read_from_file"""
    super(LMDBDatasetTest, self).setUp()
    # Copy database out because we need the path to be writable to use locks.
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_lmdb", "data.mdb")
    self.db_path = os.path.join(self.get_temp_dir(), "data.mdb")
    shutil.copy(path, self.db_path)

    filename = self.db_path

    dataset = lmdb_io.LMDBDataset([filename], batch=3)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(0, 9, 3):
        k = [
            str(i).encode(),
            str(i + 1).encode(),
            str(i + 2).encode()]
        v = [
            str(chr(ord("a") + i)).encode(),
            str(chr(ord("a") + i + 1)).encode(),
            str(chr(ord("a") + i + 2)).encode()]
        self.assertAllEqual((k, v), sess.run(get_next))
      self.assertAllEqual(
          ([str(9).encode()], [str('j').encode()]), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
