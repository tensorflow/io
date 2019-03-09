# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Kinesis Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow

from tensorflow import dtypes
from tensorflow.compat.v1 import data
from tensorflow_io import _load_library
kinesis_ops = _load_library('_kinesis_ops.so')

class KinesisDataset(data.Dataset):
  """A Kinesis Dataset that consumes the message.

  Kinesis is a managed service provided by AWS for data streaming.
  This dataset reads messages from Kinesis with each message presented
  as a `tf.string`.

  For example, we can construct and use the KinesisDataset as follows:
  ```python
  dataset = KinesisDataset(
      "kinesis_stream_name", read_indefinitely=False)
  next = dataset.make_one_shot_iterator().get_next()
  with tf.Session() as sess:
    while True:
      try:
        print(sess.run(nxt))
      except tf.errors.OutOfRangeError:
        break
  ```

  Since Kinesis is a data streaming service, data may not be available
  at the time it is being read. The argument `read_indefinitely` is
  used to control the behavior in this situation. If `read_indefinitely`
  is `True`, then `KinesisDataset` will keep retrying to retrieve data
  from the stream. If `read_indefinitely` is `False`, an `OutOfRangeError`
  is returned immediately instead.
  """

  def __init__(self,
               stream,
               shard="",
               read_indefinitely=True,
               interval=100000):
    """Create a KinesisDataset.

    Args:
      stream: A `tf.string` tensor containing the name of the stream.
      shard: A `tf.string` tensor containing the id of the shard.
      read_indefinitely: If `True`, the Kinesis dataset will keep retry
        again on `EOF` after the `interval` period. If `False`, then
        the dataset will stop on `EOF`. The default value is `True`.
      interval: The interval for the Kinesis Client to wait before
        it tries to get records again (in millisecond).
    """
    self._stream = tensorflow.convert_to_tensor(
        stream, dtype=dtypes.string, name="stream")
    self._shard = tensorflow.convert_to_tensor(
        shard, dtype=dtypes.string, name="shard")
    self._read_indefinitely = tensorflow.convert_to_tensor(
        read_indefinitely, dtype=dtypes.bool, name="read_indefinitely")
    self._interval = tensorflow.convert_to_tensor(
        interval, dtype=dtypes.int64, name="interval")
    super(KinesisDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return kinesis_ops.kinesis_dataset(
        self._stream, self._shard, self._read_indefinitely, self._interval)

  @property
  def output_classes(self):
    return tensorflow.Tensor

  @property
  def output_shapes(self):
    return tensorflow.TensorShape([])

  @property
  def output_types(self):
    return dtypes.string
