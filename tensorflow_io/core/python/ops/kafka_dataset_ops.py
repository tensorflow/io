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
"""KafkaDataset"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class KafkaIODataset(tf.data.Dataset):
    """KafkaIODataset"""

    def __init__(
        self, topic, partition, start, stop, servers, configuration, internal=True
    ):
        """Creates a `KafkaIODataset` from kafka server with an offset range.

        Args:
          topic: A `tf.string` tensor containing topic subscription.
          partition: A `tf.int64` tensor containing the partition, by default 0.
          start: A `tf.int64` tensor containing the start offset, by default 0.
          stop: A `tf.int64` tensor containing the end offset, by default -1.
          servers: An optional list of bootstrap servers, by default
             `localhost:9092`.
          configuration: An optional `tf.string` tensor containing
            configurations in [Key=Value] format.
            Global configuration: please refer to 'Global configuration properties'
              in librdkafka doc. Examples include
              ["enable.auto.commit=false", "heartbeat.interval.ms=2000"]
            Topic configuration: please refer to 'Topic configuration properties'
              in librdkafka doc. Note all topic configurations should be
              prefixed with `conf.topic.`. Examples include
              ["conf.topic.auto.offset.reset=earliest"]
            Reference: https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
          internal: Whether the dataset is being created from within the named scope.
            Default: True
        """
        with tf.name_scope("KafkaIODataset"):
            assert internal

            metadata = list(configuration or [])
            if servers is not None:
                metadata.append("bootstrap.servers=%s" % servers)
            resource = core_ops.io_kafka_readable_init(
                topic, partition, offset=0, metadata=metadata
            )
            start, stop = core_ops.io_kafka_readable_spec(resource, start, stop)

            self._resource = resource
            self._start, self._stop = start, stop

            step = 1024
            indices_start = tf.data.Dataset.range(0, stop, step)
            indices_stop = indices_start.skip(1).concatenate(
                tf.data.Dataset.from_tensor_slices([stop])
            )
            dataset = tf.data.Dataset.zip((indices_start, indices_stop))

            def f(start, stop):
                return core_ops.io_kafka_readable_read(
                    self._resource, start=start, stop=stop
                )

            dataset = dataset.map(f)
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


class KafkaStreamIODataset(tf.data.Dataset):
    """KafkaStreamIODataset"""

    def __init__(self, topic, partition, offset, servers, configuration, internal=True):
        """Creates a `StreamIODataset` from kafka server with only a start offset.

        Args:
          topic: A `tf.string` tensor containing topic subscription.
          partition: A `tf.int64` tensor containing the partition.
          offset: A `tf.int64` tensor containing the start offset.
          servers: An optional list of bootstrap servers.
             For example: `localhost:9092`.
          configuration: An optional `tf.string` tensor containing
            configurations in [Key=Value] format.
            Global configuration: please refer to 'Global configuration properties'
              in librdkafka doc. Examples include
              ["enable.auto.commit=false", "heartbeat.interval.ms=2000"]
            Topic configuration: please refer to 'Topic configuration properties'
              in librdkafka doc. Note all topic configurations should be
              prefixed with `conf.topic.`. Examples include
              ["conf.topic.auto.offset.reset=earliest"]
            Reference: https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
          internal: Whether the dataset is being created from within the named scope.
            Default: True
        """
        with tf.name_scope("KafkaStreamIODataset"):
            assert internal

            metadata = list(configuration or [])
            if servers is not None:
                metadata.append("bootstrap.servers=%s" % servers)
            resource = core_ops.io_kafka_readable_init(
                topic, partition, offset, metadata=metadata
            )

            self._resource = resource

            dataset = tf.data.experimental.Counter()
            dataset = dataset.map(
                lambda i: core_ops.io_kafka_readable_next(self._resource, i)
            )
            dataset = dataset.apply(
                tf.data.experimental.take_while(
                    lambda v: tf.greater(tf.shape(v.message)[0], 0)
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
