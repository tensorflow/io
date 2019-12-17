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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import tensorflow as tf
from tensorflow import dtypes
from tensorflow.compat.v1 import data
from tensorflow_io.core.python.ops import core_ops

decode_avro_init = core_ops.io_decode_avro_init
decode_avro = core_ops.io_decode_avro
encode_avro = core_ops.io_encode_avro

warnings.warn(
    "implementation of existing tensorflow_io.kafka.KafkaDataset is "
    "deprecated and will be replaced with the implementation in "
    "tensorflow_io.core.python.ops.kafka_dataset_ops.KafkaDataset, "
    "please check the doc of new implementation for API changes",
    DeprecationWarning)


class KafkaDataset(data.Dataset):
  """A Kafka Dataset that consumes the message.
  """

  def __init__(self,
               topics,
               servers="localhost",
               group="",
               eof=False,
               timeout=1000,
               config_global=None,
               config_topic=None,
               message_key=False):
    """Create a KafkaReader.

    Args:
      topics: A `tf.string` tensor containing one or more subscriptions,
              in the format of [topic:partition:offset:length],
              by default length is -1 for unlimited.
      servers: A list of bootstrap servers.
      group: The consumer group id.
      eof: If True, the kafka reader will stop on EOF.
      timeout: The timeout value for the Kafka Consumer to wait
               (in millisecond).
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
    """
    self._topics = tf.convert_to_tensor(
        topics, dtype=dtypes.string, name="topics")
    self._servers = tf.convert_to_tensor(
        servers, dtype=dtypes.string, name="servers")
    self._group = tf.convert_to_tensor(
        group, dtype=dtypes.string, name="group")
    self._eof = tf.convert_to_tensor(eof, dtype=dtypes.bool, name="eof")
    self._timeout = tf.convert_to_tensor(
        timeout, dtype=dtypes.int64, name="timeout")
    config_global = config_global if config_global else []
    self._config_global = tf.convert_to_tensor(
        config_global, dtype=dtypes.string, name="config_global")
    config_topic = config_topic if config_topic else []
    self._config_topic = tf.convert_to_tensor(
        config_topic, dtype=dtypes.string, name="config_topic")
    self._message_key = message_key
    super(KafkaDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return core_ops.io_kafka_dataset(self._topics, self._servers,
                                     self._group, self._eof, self._timeout,
                                     self._config_global, self._config_topic,
                                     self._message_key)

  @property
  def output_classes(self):
    return (tf.Tensor) if not self._message_key else (tf.Tensor, tf.Tensor)

  @property
  def output_shapes(self):
    return (
        tf.TensorShape([])) if not self._message_key else (
            tf.TensorShape([]), tf.TensorShape([]))

  @property
  def output_types(self):
    return (
        dtypes.string) if not self._message_key else (
            dtypes.string, dtypes.string)

def write_kafka(message,
                topic,
                servers="localhost",
                name=None):
  """
  Args:
      message: A `Tensor` of type `string`. 0-D.
      topic: A `tf.string` tensor containing one subscription,
        in the format of topic:partition.
      servers: A list of bootstrap servers.
      name: A name for the operation (optional).
  Returns:
      A `Tensor` of type `string`. 0-D.
  """
  return core_ops.io_write_kafka(
      message=message, topic=topic, servers=servers, name=name)
