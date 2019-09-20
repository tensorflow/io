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
from tensorflow_io.core.python.ops import core_golang_ops

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
      resource = core_golang_ops.prometheus_indexable_init(
          query, metadata=metadata,
          container=scope, shared_name="%s/%s" % (query, uuid.uuid4().hex))
      index_shape, index_dtype = core_golang_ops.prometheus_indexable_spec(
          resource, "index")
      value_shape, value_dtype = core_golang_ops.prometheus_indexable_spec(
          resource, "value")
      spec = tuple([tf.TensorSpec(tf.TensorShape(index_shape),
                                  tf.as_dtype(index_dtype.numpy())),
                    tf.TensorSpec(tf.TensorShape(value_shape),
                                  tf.as_dtype(value_dtype.numpy()))])
      super(PrometheusIOTensor, self).__init__(
          spec, resource, core_golang_ops.prometheus_indexable_read,
          internal=internal)
