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
"""KafkaIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops

class _KafkaIOTensorFunction(object):
  """_KafkaIOTensorFunction"""
  def __init__(self, resource, capacity):
    self._resource = resource
    self._capacity = capacity
    self._index = 0
  def __call__(self):
    items = core_ops.io_kafka_readable_read(
        self._resource,
        start=self._index, stop=self._index+self._capacity,
        shape=tf.TensorShape([None]), dtype=tf.string)
    self._index += items.shape[0]
    return items

class KafkaIOTensor(io_tensor_ops.BaseIOTensor): # pylint: disable=protected-access
  """KafkaIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               topic,
               partition,
               offset,
               tail,
               servers,
               configuration,
               internal=True):
    """KafkaIOTensor."""
    with tf.name_scope("KafkaIOTensor") as scope:
      subscription = "%s:%d:%d:%d" % (topic, partition, offset, tail)
      metadata = [e for e in configuration or []]
      if servers is not None:
        metadata.append("bootstrap.servers=%s" % servers)
      resource = core_ops.io_kafka_readable_init(
          subscription, metadata=metadata,
          container=scope,
          shared_name="%s/%s" % (subscription, uuid.uuid4().hex))
      function = _KafkaIOTensorFunction(resource, capacity=4096)
      function = io_tensor_ops._IOTensorIterablePartitionedFunction( # pylint: disable=protected-access
          function, tf.TensorShape([None]))
      spec = tf.TensorSpec(tf.TensorShape([None]), tf.string)
      super(KafkaIOTensor, self).__init__(
          spec, function, internal=internal)
