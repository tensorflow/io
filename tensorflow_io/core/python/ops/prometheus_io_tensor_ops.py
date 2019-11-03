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
"""PrometheusIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import golang_ops

class _PrometheusIOTensorFunction(object):
  """_AudioIOTensorFunction"""
  def __init__(self, function, resource, component, shape, dtype):
    self._function = function
    self._resource = resource
    self._component = component
    self._length = shape[0]
    self._shape = tf.TensorShape([None]).concatenate(shape[1:])
    self._dtype = dtype
  def __call__(self, start, stop):
    start, stop, _ = slice(start, stop).indices(self._length)
    if start >= self._length:
      raise IndexError("index %s is out of range" % slice(start, stop))
    return self._function(
        self._resource,
        start=start, stop=stop,
        component=self._component,
        shape=self._shape, dtype=self._dtype)
  @property
  def length(self):
    return self._length

class PrometheusIOTensor(io_tensor_ops._SeriesIOTensor): # pylint: disable=protected-access
  """PrometheusIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               query,
               endpoint=None,
               internal=False):
    with tf.name_scope("PrometheusIOTensor") as scope:
      metadata = [] if endpoint is None else ["endpoint: %s" % endpoint]
      resource = golang_ops.io_prometheus_readable_init(
          query, metadata=metadata,
          container=scope, shared_name="%s/%s" % (query, uuid.uuid4().hex))
      index_shape, index_dtype = golang_ops.io_prometheus_readable_spec(
          resource, "index")
      value_shape, value_dtype = golang_ops.io_prometheus_readable_spec(
          resource, "value")
      index_shape = tf.TensorShape(index_shape.numpy())
      index_dtype = tf.as_dtype(index_dtype.numpy())
      value_shape = tf.TensorShape(value_shape.numpy())
      value_dtype = tf.as_dtype(value_dtype.numpy())
      spec = tuple([tf.TensorSpec(index_shape, index_dtype),
                    tf.TensorSpec(value_shape, value_dtype)])
      index = io_tensor_ops.BaseIOTensor(
          spec[0],
          _PrometheusIOTensorFunction(
              golang_ops.io_prometheus_readable_read,
              resource, "index", index_shape, index_dtype),
          internal=internal)
      value = io_tensor_ops.BaseIOTensor(
          spec[1],
          _PrometheusIOTensorFunction(
              golang_ops.io_prometheus_readable_read,
              resource, "value", value_shape, value_dtype),
          internal=internal)
      super(PrometheusIOTensor, self).__init__(
          spec, index, value, internal=internal)
