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

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.ops import io_dataset_ops

class _KafkaIODatasetFunction():
  def __init__(self, resource):
    self._resource = resource
  def __call__(self, start, stop):
    return core_ops.io_kafka_readable_read(
        self._resource, start=start, stop=stop,
        shape=tf.TensorShape([None]), dtype=tf.string)

class KafkaIODataset(tf.data.Dataset):
  """KafkaIODataset"""

  def __init__(self,
               topic,
               partition,
               start, stop,
               servers, configuration,
               internal=True):
    """KafkaIODataset."""
    with tf.name_scope("KafkaIODataset"):
      assert internal

      metadata = list(configuration or [])
      if servers is not None:
        metadata.append("bootstrap.servers=%s" % servers)
      resource = core_ops.io_kafka_readable_init_v(
          topic, partition, start, stop, metadata=metadata)
      start, stop = core_ops.io_kafka_readable_spec_v(resource)

      self._resource = resource

      step = 1024
      indices_start = tf.data.Dataset.range(0, stop, step)
      indices_stop = indices_start.skip(1).concatenate(
          tf.data.Dataset.from_tensor_slices([stop]))
      dataset = tf.data.Dataset.zip((indices_start, indices_stop))
      def f(start, stop):
        return core_ops.io_kafka_readable_read_v(
            self._resource, start=start, stop=stop)
      dataset = dataset.map(f)
      dataset = dataset.unbatch()

      self._dataset = dataset
      super(KafkaIODataset, self).__init__(self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec

class KafkaStreamIODataset(io_dataset_ops._StreamIODataset): # pylint: disable=protected-access
  """KafkaStreamIODataset"""

  def __init__(self,
               topic,
               partition,
               offset,
               servers,
               configuration,
               internal=True):
    """KafkaStreamIODataset."""
    with tf.name_scope("KafkaStreamIODataset") as scope:
      subscription = "%s:%d:%d" % (topic, partition, offset)
      metadata = list(configuration or [])
      if servers is not None:
        metadata.append("bootstrap.servers=%s" % servers)
      resource = core_ops.io_kafka_readable_init(
          subscription, metadata=metadata,
          container=scope,
          shared_name="%s/%s" % (subscription, uuid.uuid4().hex))

      super(KafkaStreamIODataset, self).__init__(
          _KafkaIODatasetFunction(resource), internal=internal)
