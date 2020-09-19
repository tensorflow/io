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

import tensorflow as tf

from tensorflow_io.core.python.ops import core_ops


class KinesisIODataset(tf.data.Dataset):
    """A Kinesis Dataset that consumes the message.

    Kinesis is a managed service provided by AWS for data streaming.
    This dataset reads messages from Kinesis with each message presented
    as a `tf.string`.

    For example, we can construct and use the KinesisIODataset as follows:
    ```python
    dataset = KinesisIODataset(
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
    is `True`, then `KinesisIODataset` will keep retrying to retrieve data
    from the stream. If `read_indefinitely` is `False`, an `OutOfRangeError`
    is returned immediately instead.
    """

    def __init__(self, stream, shard="", internal=False):
        """Create a KinesisIODataset.

        Args:
          stream: A `tf.string` tensor containing the name of the stream.
          shard: A `tf.string` tensor containing the id of the shard.
        """
        with tf.name_scope("KinesisIODataset"):
            assert internal

            metadata = []
            metadata.append("shard=%s" % shard)
            resource = core_ops.io_kinesis_readable_init(stream, metadata)

            self._resource = resource

            dataset = tf.data.experimental.Counter()
            dataset = dataset.map(
                lambda _: core_ops.io_kinesis_readable_read(self._resource)
            )
            dataset = dataset.apply(
                tf.data.experimental.take_while(
                    lambda v: tf.greater(tf.shape(v.data)[0], 0)
                )
            )
            dataset = dataset.unbatch()

            self._dataset = dataset
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
