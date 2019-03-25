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
NOTE: boto3 is needed and the following has to run first:
```
$ bash -x kinesis_test.sh start kinesis
```
After test has been completed, run:
```
$ bash -x kinesis_test.sh stop kinesis
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pytest
import boto3

import tensorflow
tensorflow.compat.v1.disable_eager_execution()

from tensorflow import dtypes              # pylint: disable=wrong-import-position
from tensorflow import errors              # pylint: disable=wrong-import-position
from tensorflow import test                # pylint: disable=wrong-import-position
from tensorflow.compat.v1 import data      # pylint: disable=wrong-import-position

import tensorflow_io.kinesis as kinesis_io # pylint: disable=wrong-import-position

if sys.platform == "darwin":
  pytest.skip("kinesis/localstack is failing on macOS", allow_module_level=True)

class KinesisDatasetTest(test.TestCase):
  """Tests for KinesisDataset."""

  def test_kinesis_dataset_one_shard(self):
    """Tests for KinesisDataset."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'ACCESS_KEY'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'SECRET_KEY'
    os.environ['KINESIS_USE_HTTPS'] = '0'
    os.environ['KINESIS_ENDPOINT'] = 'localhost:4568'

    client = boto3.client(
        'kinesis',
        region_name='us-east-1',
        endpoint_url='http://localhost:4568')

    # Setup the Kinesis with 1 shard.
    stream_name = "tf_kinesis_test_1"
    client.create_stream(StreamName=stream_name, ShardCount=1)
    # Wait until stream exists, default is 10 * 18 seconds.
    client.get_waiter('stream_exists').wait(StreamName=stream_name)
    for i in range(10):
      data_v = "D" + str(i)
      client.put_record(
          StreamName=stream_name,
          Data=data_v,
          PartitionKey="TensorFlow" + str(i))

    stream = tensorflow.compat.v1.placeholder(dtypes.string, shape=[])
    num_epochs = tensorflow.compat.v1.placeholder(dtypes.int64, shape=[])
    batch_size = tensorflow.compat.v1.placeholder(dtypes.int64, shape=[])

    repeat_dataset = kinesis_io.KinesisDataset(
        stream, read_indefinitely=False).repeat(num_epochs)
    batch_dataset = repeat_dataset.batch(batch_size)

    iterator = data.Iterator.from_structure(batch_dataset.output_types)
    init_op = iterator.make_initializer(repeat_dataset)
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      # Basic test: read from shard 0 of stream 1.
      sess.run(init_op, feed_dict={stream: stream_name, num_epochs: 1})
      for i in range(10):
        self.assertEqual(("D" + str(i)).encode(), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

    client.delete_stream(StreamName=stream_name)
    # Wait until stream deleted, default is 10 * 18 seconds.
    client.get_waiter('stream_not_exists').wait(StreamName=stream_name)

  def test_kinesis_dataset_two_shards(self):
    """Tests for KinesisDataset."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'ACCESS_KEY'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'SECRET_KEY'
    os.environ['KINESIS_USE_HTTPS'] = '0'
    os.environ['KINESIS_ENDPOINT'] = 'localhost:4568'

    client = boto3.client(
        'kinesis',
        region_name='us-east-1',
        endpoint_url='http://localhost:4568')

    # Setup the Kinesis with 2 shards.
    stream_name = "tf_kinesis_test_2"
    client.create_stream(StreamName=stream_name, ShardCount=2)
    # Wait until stream exists, default is 10 * 18 seconds.
    client.get_waiter('stream_exists').wait(StreamName=stream_name)

    for i in range(10):
      data_v = "D" + str(i)
      client.put_record(
          StreamName=stream_name,
          Data=data_v,
          PartitionKey="TensorFlow" + str(i))
    response = client.describe_stream(StreamName=stream_name)
    shard_id_0 = response["StreamDescription"]["Shards"][0]["ShardId"]
    shard_id_1 = response["StreamDescription"]["Shards"][1]["ShardId"]

    stream = tensorflow.compat.v1.placeholder(dtypes.string, shape=[])
    shard = tensorflow.compat.v1.placeholder(dtypes.string, shape=[])
    num_epochs = tensorflow.compat.v1.placeholder(dtypes.int64, shape=[])
    batch_size = tensorflow.compat.v1.placeholder(dtypes.int64, shape=[])

    repeat_dataset = kinesis_io.KinesisDataset(
        stream, shard, read_indefinitely=False).repeat(num_epochs)
    batch_dataset = repeat_dataset.batch(batch_size)

    iterator = data.Iterator.from_structure(batch_dataset.output_types)
    init_op = iterator.make_initializer(repeat_dataset)
    get_next = iterator.get_next()

    data_v = list()
    with self.cached_session() as sess:
      # Basic test: read from shard 0 of stream 2.
      sess.run(
          init_op, feed_dict={
              stream: stream_name, shard: shard_id_0, num_epochs: 1})
      with self.assertRaises(errors.OutOfRangeError):
        # Use range(11) to guarantee the OutOfRangeError.
        for i in range(11):
          data_v.append(sess.run(get_next))

      # Basic test: read from shard 1 of stream 2.
      sess.run(
          init_op, feed_dict={
              stream: stream_name, shard: shard_id_1, num_epochs: 1})
      with self.assertRaises(errors.OutOfRangeError):
        # Use range(11) to guarantee the OutOfRangeError.
        for i in range(11):
          data_v.append(sess.run(get_next))

    data_v.sort()
    self.assertEqual(data_v, [("D" + str(i)).encode() for i in range(10)])

    client.delete_stream(StreamName=stream_name)
    # Wait until stream deleted, default is 10 * 18 seconds.
    client.get_waiter('stream_not_exists').wait(StreamName=stream_name)


if __name__ == "__main__":
  test.main()
