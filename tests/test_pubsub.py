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
"""Tests for PubSubDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import pytest
from google.cloud import pubsub_v1

import tensorflow
tensorflow.compat.v1.disable_eager_execution()

from tensorflow import dtypes         # pylint: disable=wrong-import-position
from tensorflow import errors         # pylint: disable=wrong-import-position
from tensorflow import test           # pylint: disable=wrong-import-position
from tensorflow.compat.v1 import data # pylint: disable=wrong-import-position

from tensorflow_io.pubsub.python.ops import pubsub_dataset_ops # pylint: disable=wrong-import-position

if sys.platform == "darwin":
  pytest.skip("pubsub is not supported on macOS yet", allow_module_level=True)

class PubSubDatasetTest(test.TestCase):
  """Tests for PubSubDataset."""

  # The Pubsub server has to be setup before the test
  # and tear down after the test manually.
  # The docker engine has to be installed.
  #
  # To setup the Pubsub server:
  # $ bash pubsub_test.sh start pubsub
  #
  # To team down the Pubsub server:
  # $ bash pubsub_test.sh stop pubsub
  def setUp(self): # pylint: disable=invalid-name
    """setUp"""
    super(PubSubDatasetTest, self).setUp()

    self._channel = "e{}e".format(time.time())

    os.environ['PUBSUB_EMULATOR_HOST'] = 'localhost:8085'
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(
        "pubsub-project", "pubsub_topic_"+self._channel)
    publisher.create_topic(topic_path)
    # print('Topic created: {}'.format(topic))
    subscriber = pubsub_v1.SubscriberClient()
    for i in range(4):
      subscription_path = subscriber.subscription_path(
          "pubsub-project",
          "pubsub_subscription_{}_{}".format(self._channel, i))
      subscription = subscriber.create_subscription(
          subscription_path, topic_path)
      print('Subscription created: {}'.format(subscription))
    for n in range(0, 10):
      data_v = u'Message number {}'.format(n)
      # Data must be a bytestring
      data_v = data_v.encode('utf-8')
      # When you publish a message, the client returns a future.
      future = publisher.publish(topic_path, data=data_v)
      print('Published {} of message ID {}.'.format(data_v, future.result()))
    print('Published messages.')

  def test_pubsub_dataset(self):
    """Tests for PubSubDataset."""
    subscriptions = tensorflow.compat.v1.placeholder(
        dtypes.string, shape=[None])
    num_epochs = tensorflow.compat.v1.placeholder(
        dtypes.int64, shape=[])
    batch_size = tensorflow.compat.v1.placeholder(
        dtypes.int64, shape=[])

    repeat_dataset = pubsub_dataset_ops.PubSubDataset(
        subscriptions,
        server="http://localhost:8085",
        eof=True).repeat(num_epochs)
    batch_dataset = repeat_dataset.batch(batch_size)

    iterator = data.Iterator.from_structure(batch_dataset.output_types)
    init_op = iterator.make_initializer(repeat_dataset)
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      # Basic test: read from subscription 0.
      sess.run(init_op, feed_dict={
          subscriptions: [
              "projects/pubsub-project/"
              "subscriptions/pubsub_subscription_"+self._channel+"_0"],
          num_epochs: 1})
      for i in range(10):
        self.assertEqual(
            ("Message number " + str(i)).encode(),
            sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Basic test: read from subscription 1.
      sess.run(init_op, feed_dict={
          subscriptions: [
              "projects/pubsub-project/"
              "subscriptions/pubsub_subscription_"+self._channel+"_1"],
          num_epochs: 1})
      for i in range(10):
        self.assertEqual(
            ("Message number " + str(i)).encode(),
            sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Basic test: read from both subscriptions.
      sess.run(
          init_op,
          feed_dict={
              subscriptions: [
                  "projects/pubsub-project/"
                  "subscriptions/pubsub_subscription_"+self._channel+"_2",
                  "projects/pubsub-project/"
                  "subscriptions/pubsub_subscription_"+self._channel+"_3"],
              num_epochs: 1})
      for _ in range(2):
        for i in range(10):
          self.assertEqual(
              ("Message number " + str(i)).encode(),
              sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
