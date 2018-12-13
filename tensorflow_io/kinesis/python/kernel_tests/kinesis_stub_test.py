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
"""Tests for KinesisDataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test

from tensorflow_io.kinesis.python.ops import kinesis_dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops


class KinesisDatasetStubTest(test.TestCase):
  """Tests for KinesisDataset."""

  def test_kinesis_dataset_stub(self):
    """Stub Tests for KinesisDataset, no session"""
    stream = array_ops.placeholder(dtypes.string, shape=[])
    num_epochs = array_ops.placeholder(dtypes.int64, shape=[])
    batch_size = array_ops.placeholder(dtypes.int64, shape=[])

    repeat_dataset = kinesis_dataset_ops.KinesisDataset(
        stream, read_indefinitely=False).repeat(num_epochs)
    batch_dataset = repeat_dataset.batch(batch_size)

    iterator = iterator_ops.Iterator.from_structure(batch_dataset.output_types)
    init_op = iterator.make_initializer(repeat_dataset)
    get_next = iterator.get_next()


if __name__ == "__main__":
  test.main()
