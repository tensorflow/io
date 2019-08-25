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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

class KafkaDataset(tf.compat.v2.data.Dataset):
  """KafkaDataset"""

  def __init__(self,
               subscription,
               servers=None,
               configuration=None):
    """Create a KafkaDataset.
    Args:
      subscription: A `tf.string` tensor containing subscription,
        in the format of [topic:partition:offset:length],
        by default length is -1 for unlimited.
      servers: A list of bootstrap servers, by default `localhost:9092`.
      configuration: A `tf.string` tensor containing configurations
        in [Key=Value] format. There are three types of configurations,
        Global configuration: please refer to 'Global configuration properties'
          in librdkafka doc. Examples include
          ["enable.auto.commit=false", "heartbeat.interval.ms=2000"]
        Topic configuration: please refer to 'Topic configuration properties'
          in librdkafka doc. Note all topic configurations should be
          prefixed with `configuration.topic.`. Examples include
          ["conf.topic.auto.offset.reset=earliest"]
        Dataset configuration: there are two configurations available,
          `conf.eof=0|1`: if True, the KafkaDaset will stop on EOF (default).
          `conf.timeout=milliseconds`: timeout value for Kafka Consumer to wait.
    """
    with tf.name_scope("KafkaDataset") as scope:
      metadata = [e for e in configuration or []]
      if servers is not None:
        metadata.append("bootstrap.servers=%s" % servers)
      resource, _, _ = core_ops.kafka_iterable_init(
          subscription, metadata=metadata,
          container=scope,
          shared_name="%s/%s" % (subscription, uuid.uuid4().hex))

      capacity = 4096
      dataset = tf.compat.v2.data.Dataset.range(0, sys.maxsize, capacity)
      dataset = dataset.map(
          lambda i: core_ops.kafka_iterable_next(
              resource, capacity, component=0,
              dtype=tf.string, shape=tf.TensorShape([None])))
      dataset = dataset.apply(
          tf.data.experimental.take_while(
              lambda v: tf.greater(tf.shape(v)[0], 0)))
      dataset = dataset.unbatch()

      self._resource = resource
      self._dataset = dataset
      super(KafkaDataset, self).__init__(
          self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec
