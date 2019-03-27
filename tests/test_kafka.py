# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for KafkaDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import pytest

import tensorflow
tensorflow.compat.v1.disable_eager_execution()

from tensorflow import dtypes          # pylint: disable=wrong-import-position
from tensorflow import errors          # pylint: disable=wrong-import-position
from tensorflow import test            # pylint: disable=wrong-import-position
from tensorflow.compat.v1 import data  # pylint: disable=wrong-import-position

import tensorflow_io.kafka as kafka_io # pylint: disable=wrong-import-position

if sys.platform == "darwin":
  pytest.skip("kafka is failing on macOS", allow_module_level=True)

class KafkaDatasetTest(test.TestCase):
  """Tests for KafkaDataset."""

  # The Kafka server has to be setup before the test
  # and tear down after the test manually.
  # The docker engine has to be installed.
  #
  # To setup the Kafka server:
  # $ bash kafka_test.sh start kafka
  #
  # To team down the Kafka server:
  # $ bash kafka_test.sh stop kafka

  def test_kafka_dataset(self):
    """Tests for KafkaDataset."""
    topics = tensorflow.compat.v1.placeholder(dtypes.string, shape=[None])
    num_epochs = tensorflow.compat.v1.placeholder(dtypes.int64, shape=[])
    batch_size = tensorflow.compat.v1.placeholder(dtypes.int64, shape=[])

    repeat_dataset = kafka_io.KafkaDataset(
        topics, group="test", eof=True).repeat(num_epochs)
    batch_dataset = repeat_dataset.batch(batch_size)

    iterator = data.Iterator.from_structure(batch_dataset.output_types)
    init_op = iterator.make_initializer(repeat_dataset)
    init_batch_op = iterator.make_initializer(batch_dataset)
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      # Basic test: read from topic 0.
      sess.run(init_op, feed_dict={topics: ["test:0:0:4"], num_epochs: 1})
      for i in range(5):
        self.assertEqual(("D" + str(i)).encode(), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Basic test: read from topic 1.
      sess.run(init_op, feed_dict={topics: ["test:0:5:-1"], num_epochs: 1})
      for i in range(5):
        self.assertEqual(("D" + str(i + 5)).encode(), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Basic test: read from both topics.
      sess.run(
          init_op,
          feed_dict={
              topics: ["test:0:0:4", "test:0:5:-1"],
              num_epochs: 1
          })
      for j in range(2):
        for i in range(5):
          self.assertEqual(("D" + str(i + j * 5)).encode(), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Test repeated iteration through both files.
      sess.run(
          init_op,
          feed_dict={
              topics: ["test:0:0:4", "test:0:5:-1"],
              num_epochs: 10
          })
      for _ in range(10):
        for j in range(2):
          for i in range(5):
            self.assertEqual(
                ("D" + str(i + j * 5)).encode(),
                sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Test batched and repeated iteration through both files.
      sess.run(
          init_batch_op,
          feed_dict={
              topics: ["test:0:0:4", "test:0:5:-1"],
              num_epochs: 10,
              batch_size: 5
          })
      for _ in range(10):
        self.assertAllEqual([("D" + str(i)).encode() for i in range(5)],
                            sess.run(get_next))
        self.assertAllEqual([("D" + str(i + 5)).encode() for i in range(5)],
                            sess.run(get_next))

  def test_write_kafka(self):
    """test_write_kafka"""
    channel = "e{}e".format(time.time())

    # Start with reading test topic, replace `D` with `e(time)e`,
    # and write to test_e(time)e` topic.
    dataset = kafka_io.KafkaDataset(
        topics=["test:0:0:4"], group="test", eof=True)
    dataset = dataset.map(
        lambda x: kafka_io.write_kafka(
            tensorflow.strings.regex_replace(x, "D", channel),
            topic="test_"+channel))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      # Basic test: read from topic 0.
      sess.run(init_op)
      for i in range(5):
        self.assertEqual((channel + str(i)).encode(), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

    # Reading from `test_e(time)e` we should get the same result
    dataset = kafka_io.KafkaDataset(
        topics=["test_"+channel], group="test", eof=True)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(5):
        self.assertEqual((channel + str(i)).encode(), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

if __name__ == "__main__":
  test.main()
