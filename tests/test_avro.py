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
"""Tests for AvroDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow import dtypes  # pylint: disable=wrong-import-position
from tensorflow import errors  # pylint: disable=wrong-import-position
from tensorflow import test    # pylint: disable=wrong-import-position
from tensorflow.compat.v1 import data # pylint: disable=wrong-import-position

import tensorflow_io.avro as avro_io # pylint: disable=wrong-import-position

class AvroDatasetTest(test.TestCase):
  """AvroDatasetTest"""

  def test_avro_dataset(self):
    """Test case for AvroDataset."""
    # The test.bin was created from avro/lang/c++/examples/datafile.cc.
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_avro", "test.bin")
    filename = "file://" + filename

    schema_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_avro", "cpx.json")
    with open(schema_filename, 'r') as f:
      schema = f.read()

    columns = ['im', 're']
    output_types = (dtypes.float64, dtypes.float64)
    num_repeats = 2

    dataset = avro_io.AvroDataset(
        [filename], columns, schema, output_types).repeat(num_repeats)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):
        for i in range(100):
          (im, re) = (i + 100, i * 100)
          vv = sess.run(get_next)
          self.assertAllClose((im, re), vv)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

    dataset = avro_io.AvroDataset(
        [filename, filename], columns, schema, output_types, batch=3)
    iterator = data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for ii in range(0, 198, 3):
        i = ii % 100
        (im, re) = (
            [i + 100, ((i + 1) % 100) + 100, ((i + 2) % 100) + 100],
            [i * 100, ((i + 1) % 100) * 100, ((i + 2) % 100) * 100])
        vv = sess.run(get_next)
        self.assertAllClose((im, re), vv)
      (im, re) = ([198, 199], [9800, 9900])
      vv = sess.run(get_next)
      self.assertAllClose((im, re), vv)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

if __name__ == "__main__":
  test.main()
