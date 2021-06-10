# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Kafka Dataset."""

import warnings

import tensorflow as tf
from tensorflow import dtypes
from tensorflow.compat.v1 import data
from tensorflow_io.python.ops import core_ops
from tensorflow_io.python.utils import deprecation

warnings.warn(
    "implementation of existing tensorflow_io.kafka.KafkaDataset is "
    "deprecated and will be replaced with the implementation in "
    "tensorflow_io.core.python.ops.kafka_dataset_ops.KafkaDataset, "
    "please check the doc of new implementation for API changes",
    DeprecationWarning,
)


@deprecation.deprecate("Use tfio.IODataset.from_kafka(...) instead")
class KafkaDataset(data.Dataset):
    """A Kafka Dataset that consumes the messages."""

    def __init__(
        self,
        topics,
        servers="localhost",
        group="",
        eof=False,
        timeout=1000,
        config_global=None,
        config_topic=None,
        message_key=False,
        message_offset=False,
    ):
        """Create a KafkaReader.

        Args:
            topics: A `tf.string` tensor containing one or more subscriptions,
                    in the format of [topic:partition:offset:length],
                    by default length is -1 for unlimited.
                    eg. ["sampleTopic:0:0:10"] will fetch the first 10 messages from
                    the 0th partition of sampleTopic.
                    eg. ["sampleTopic:0:0:10","sampleTopic:1:0:10"] will fetch
                    the first 10 messages from the 0th partition followed
                    by the first 10 messages from the 1st partition of sampleTopic.
            servers: A list of bootstrap servers.
            group: The consumer group id.
            eof: If True, the kafka reader will stop on EOF.
            timeout: The timeout value for the Kafka Consumer to wait
                    (in milliseconds).
            config_global: A `tf.string` tensor containing global configuration
                            properties in [Key=Value] format,
                            eg. ["enable.auto.commit=false",
                                "heartbeat.interval.ms=2000"],
                            please refer to 'Global configuration properties'
                            in librdkafka doc.
            config_topic: A `tf.string` tensor containing topic configuration
                            properties in [Key=Value] format,
                            eg. ["auto.offset.reset=earliest"],
                            please refer to 'Topic configuration properties'
                            in librdkafka doc.
            message_key: If True, the kafka will output both message value and key.
            message_offset: If True, the kafka will output both message value and offset,
                            the offset info like 'partition-index:offset'.
        """
        self._topics = tf.convert_to_tensor(topics, dtype=dtypes.string, name="topics")
        self._servers = tf.convert_to_tensor(
            servers, dtype=dtypes.string, name="servers"
        )
        self._group = tf.convert_to_tensor(group, dtype=dtypes.string, name="group")
        self._eof = tf.convert_to_tensor(eof, dtype=dtypes.bool, name="eof")
        self._timeout = tf.convert_to_tensor(
            timeout, dtype=dtypes.int64, name="timeout"
        )
        config_global = config_global if config_global else []
        self._config_global = tf.convert_to_tensor(
            config_global, dtype=dtypes.string, name="config_global"
        )
        config_topic = config_topic if config_topic else []
        self._config_topic = tf.convert_to_tensor(
            config_topic, dtype=dtypes.string, name="config_topic"
        )
        self._message_key = message_key
        self._message_offset = message_offset
        super().__init__()

    def _inputs(self):
        return []

    def _as_variant_tensor(self):
        return core_ops.io_kafka_dataset(
            self._topics,
            self._servers,
            self._group,
            self._eof,
            self._timeout,
            self._config_global,
            self._config_topic,
            self._message_key,
            self._message_offset,
        )

    @property
    def output_classes(self):
        if self._message_key ^ self._message_offset:
            return (tf.Tensor, tf.Tensor)
        elif self._message_key and self._message_offset:
            return (tf.Tensor, tf.Tensor, tf.Tensor)
        else:
            return tf.Tensor

    @property
    def output_shapes(self):
        if self._message_key ^ self._message_offset:
            return (tf.TensorShape([]), tf.TensorShape([]))
        elif self._message_key and self._message_offset:
            return (
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
            )
        else:
            return tf.TensorShape([])

    @property
    def output_types(self):
        if self._message_key ^ self._message_offset:
            return (dtypes.string, dtypes.string)
        elif self._message_key and self._message_offset:
            return (dtypes.string, dtypes.string, dtypes.string)
        else:
            return dtypes.string


class KafkaOutputSequence:
    """KafkaOutputSequence"""

    def __init__(self, topic, servers="localhost", configuration=None):
        """Create a `KafkaOutputSequence`.

        Args:
            topic: A `tf.string` tensor containing one subscription,
                in the format of topic:partition.
            servers: A list of bootstrap servers.
            configuration: A `tf.string` tensor containing global configuration
                            properties in [Key=Value] format,
                            eg. ["enable.auto.commit=false",
                                "heartbeat.interval.ms=2000"],
                            please refer to 'Global configuration properties'
                            in librdkafka doc.
        """
        self._topic = topic
        metadata = list(configuration or [])
        if servers is not None:
            metadata.append("bootstrap.servers=%s" % servers)
        self._resource = core_ops.io_kafka_output_sequence(
            topic=topic, metadata=metadata
        )

    def setitem(self, index, item):
        """Set an indexed item in the `KafkaOutputSequence`.

        Args:
            index: An index in the sequence. The index tensor must be a scalar.
            item: the item which is associated with the index in the output sequence.
                    The item tensor must be a scalar.
        """
        core_ops.io_kafka_output_sequence_set_item(self._resource, index, item)

    def flush(self):
        """Flush the `KafkaOutputSequence`."""
        core_ops.io_kafka_output_sequence_flush(self._resource)


def write_kafka(message, topic, servers="localhost", name=None):
    """Write messages to the kafka topic

    Args:
        message: The `tf.string` tensor containing the message
            to be written into the topic.
        topic: A `tf.string` tensor containing one subscription,
            in the format of topic:partition.
        servers: A list of bootstrap servers.
        name: A name for the operation (optional).
    Returns:
        A `Tensor` of type `string`. 0-D.
    """
    return core_ops.io_write_kafka(
        message=message, topic=topic, servers=servers, name=name
    )
