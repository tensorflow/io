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
"""KafkaBatchIODatasets"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class KafkaBatchIODataset(tf.data.Dataset):
    """Represents a streaming batch dataset from kafka using consumer groups.

    The dataset is created by fetching batches of messages from kafka using consumer clients
    which are part of a consumer group. Each batch of messages is of type `tf.data.Dataset`.
    This dataset is suitable in scenarios where the 'keys' and 'messages' in kafka topics
    are synonimous with 'labels' and 'data' items respectively. Thus, enabling the user
    to train their model in an online learning fashion.

    The dataset is similar to the `tfio.experimental.streaming.KafkaGroupIODataset` in it's
    consumer client configuration as it utilizes the consumer groups for retrieving messages
    from the topics.

    The dataset can be prepared and iterated in the following manner:

    >>> import tensorflow_io as tfio
    >>> dataset = tfio.experimental.streaming.KafkaBatchIODataset(
                        topics=["topic1"],
                        group_id="cg",
                        servers="localhost:9092"
                    )

    >>> for mini_batch in dataset:
    ...     mini_batch = mini_batch.map(
    ...            lambda m, k: (tf.cast(m, tf.float32), tf.cast(k, tf.float32)))

    Since `mini_batch` is of type `tf.data.Dataset` we can perform all the operations that it
    inherits from `tf.data.Dataset`.

    To train a keras model on this stream of incoming data:

    >>> for mini_batch in dataset:
    ...     mini_batch = mini_batch.map(
    ...            lambda m, k: (tf.cast(m, tf.float32), tf.cast(k, tf.float32)))
    ...     model.fit(mini_batch, epochs=10)

    The `mini_batch` can be directly passed into the `tf.keras` model for training.
    """

    def __init__(
        self,
        topics,
        group_id,
        servers,
        message_timeout=5000,
        stream_timeout=5000,
        configuration=None,
        internal=True,
    ):
        """
        Args:
          topics: A `tf.string` tensor containing topic names in [topic] format.
            For example: ["topic1"]
          group_id: The id of the consumer group. For example: cgstream
          servers: An optional list of bootstrap servers.
            For example: `localhost:9092`.
          message_timeout: An optional timeout value (in milliseconds) for retrieving messages
            from kafka. Default value is 5000.
          stream_timeout: An optional timeout value (in milliseconds) to wait for the new messages
            from kafka to be retrieved by the consumers. Default value is 5000.
            NOTE: The `stream_timeout` value should always be greater than or equal to the `message_timeout`.
            value.
          configuration: An optional `tf.string` tensor containing
            configurations in [Key=Value] format.
            Global configuration: please refer to 'Global configuration properties'
              in librdkafka doc. Examples include
              ["enable.auto.commit=false", "heartbeat.interval.ms=2000"]
            Topic configuration: please refer to 'Topic configuration properties'
              in librdkafka doc. Note all topic configurations should be
              prefixed with `configuration.topic.`. Examples include
              ["conf.topic.auto.offset.reset=earliest"]
          internal: Whether the dataset is being created from within the named scope.
            Default: True
        """
        with tf.name_scope("KafkaBatchIODataset"):
            assert internal

            if stream_timeout < message_timeout:
                raise ValueError(
                    "stream_timeout {} is less than the message_timeout {}".format(
                        stream_timeout, message_timeout
                    )
                )
            metadata = list(configuration or [])
            if group_id is not None:
                metadata.append("group.id=%s" % group_id)
            if servers is not None:
                metadata.append("bootstrap.servers=%s" % servers)
            resource = core_ops.io_kafka_group_readable_init(
                topics=topics, metadata=metadata
            )

            self._resource = resource
            dataset = tf.data.experimental.Counter()
            dataset = dataset.map(
                lambda i: core_ops.io_kafka_group_readable_next(
                    input=self._resource,
                    index=i,
                    message_timeout=message_timeout,
                    stream_timeout=stream_timeout,
                )
            )
            dataset = dataset.apply(
                tf.data.experimental.take_while(
                    lambda v: tf.greater(tf.shape(v.message)[0], 0)
                )
            )
            dataset = dataset.map(
                lambda v: tf.data.Dataset.zip(
                    (
                        tf.data.Dataset.from_tensor_slices(v.message),
                        tf.data.Dataset.from_tensor_slices(v.key),
                    )
                )
            )
            self._dataset = dataset
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
