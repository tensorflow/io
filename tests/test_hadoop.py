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
"""Tests for SequenceFileDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow
tensorflow.compat.v1.disable_eager_execution()

from tensorflow import dtypes
from tensorflow import errors
from tensorflow.compat.v1 import data
from tensorflow.compat.v1 import resource_loader

import tensorflow_io.hadoop as hadoop_io

from tensorflow import test

class SequenceFileDatasetTest(test.TestCase):

  def test_sequence_file_dataset(self):
    """Test case for SequenceFileDataset.

    The file is generated with `org.apache.hadoop.io.Text` for key/value.
    There are 25 records in the file with the format of:
    key = XXX
    value = VALUEXXX
    where XXX is replaced as the line number (starts with 001).
    """
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_hadoop", "string.seq")

    num_repeats = 2

    dataset = hadoop_io.SequenceFileDataset([filename]).repeat(
        num_repeats)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):  # Dataset is repeated.
        for i in range(25):  # 25 records.
          v0 = ("%03d" % (i + 1)).encode()
          v1 = ("VALUE%03d" % (i + 1)).encode()
          self.assertEqual((v0, v1), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
