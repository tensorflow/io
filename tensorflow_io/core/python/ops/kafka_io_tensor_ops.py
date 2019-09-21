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

class KafkaIOTensor(io_tensor_ops.BaseIOTensor): # pylint: disable=protected-access
  """KafkaIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               subscription,
               servers=None,
               configuration=None,
               internal=False):
    with tf.name_scope("KafkaIOTensor") as scope:
      metadata = [e for e in configuration or []]
      if servers is not None:
        metadata.append("bootstrap.servers=%s" % servers)
      iterable = core_ops.kafka_iterable_init(
          subscription, metadata=metadata,
          container=scope,
          shared_name="%s/%s" % (subscription, uuid.uuid4().hex))
      resource = core_ops.kafka_indexable_init(
          subscription, metadata=metadata, iterable=iterable,
          container=scope,
          shared_name="%s/%s" % (subscription, uuid.uuid4().hex))
      shape, dtype = core_ops.kafka_indexable_spec(resource, 0)
      spec = tf.TensorSpec(
          tf.TensorShape(shape.numpy()), tf.as_dtype(dtype.numpy()))

      class _Function(object):
        def __init__(self, func, spec):
          self._func = func
          self._shape = tf.TensorShape([None]).concatenate(spec.shape[1:])
          self._dtype = spec.dtype

        def __call__(self, resource, start, stop):
          return self._func(
              resource, start=start, stop=stop,
              shape=self._shape, dtype=self._dtype)

      self._iterable = iterable
      super(KafkaIOTensor, self).__init__(
          spec, resource,
          _Function(core_ops.kafka_indexable_read, spec),
          partitions=None, internal=internal)
