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
"""ServerDataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

class ServerDataset(tf.compat.v2.data.Dataset):
  """ServerDataset"""

  def __init__(self, server, component, spec, label):
    """Create a ServerDataset.
    Args:
      server: An IOServer.
      component: A reference component key.
    """
    with tf.name_scope("ServerDataset") as scope:
      resource = server._resource

      shape = tf.TensorShape([e if i != 0 else None for (i, e) in enumerate(spec.shape.as_list())])
      dtype = spec.dtype

      label_shape = tf.TensorShape([e if i != 0 else None for (i, e) in enumerate(label.shape.as_list())])
      label_dtype = label.dtype

      capacity = 4096
      dataset = tf.compat.v2.data.Dataset.range(0, sys.maxsize, capacity)
      dataset = dataset.map(
          lambda i: core_ops.grpcio_server_iterable_next(
              resource, capacity, component=component,
              shape=shape, dtype=dtype,
              label_shape=label_shape, label_dtype=label_dtype)).map(
                  lambda v: (v.value, v.label))

      dataset = dataset.apply(
          tf.data.experimental.take_while(
              lambda value, label: tf.greater(tf.shape(value)[0], 0)))
      dataset = dataset.unbatch()

      self._resource = resource
      self._dataset = dataset
      super(ServerDataset, self).__init__(
          self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec
