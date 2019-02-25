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

import tensorflow
tensorflow.compat.v1.disable_eager_execution()

from tensorflow import dtypes
from tensorflow import errors

import tensorflow_io.lmdb as lmdb_io

from tensorflow import test


class LMDBDatasetTest(test.TestCase):

  def setUp(self):
    super(LMDBDatasetTest, self).setUp()
    # Copy database out because we need the path to be writable to use locks.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_lmdb", "data.mdb")
    self.db_path = os.path.join(self.get_temp_dir(), "data.mdb")
    shutil.copy(path, self.db_path)

  def testReadFromFile(self):
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


if __name__ == "__main__":
  test.main()
