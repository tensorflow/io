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
"""PrometheusTimestampIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops

class PrometheusTimestampIOTensor(io_tensor_ops._ColumnIOTensor): # pylint: disable=protected-access
  """PrometheusTimestampIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               query,
               endpoint=None,
               internal=False):
    with tf.name_scope("PrometheusTimestampIOTensor") as scope:
      metadata = ["column: timestamp"]
      if endpoint is not None:
        metadata.append(["endpoint: %s" % endpoint])
      resource, dtypes, shapes, _ = core_ops.prometheus_indexable_init(
          query, metadata=metadata,
          container=scope, shared_name="%s/%s" % (query, uuid.uuid4().hex))
      super(PrometheusTimestampIOTensor, self).__init__(
          shapes, dtypes, resource, core_ops.prometheus_indexable_get_item,
          internal=internal)

class PrometheusValueIOTensor(io_tensor_ops._ColumnIOTensor): # pylint: disable=protected-access
  """PrometheusValueIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               query,
               endpoint=None,
               internal=False):
    with tf.name_scope("PrometheusTimestampIOTensor") as scope:
      metadata = ["column: value"]
      if endpoint is not None:
        metadata.append(["endpoint: %s" % endpoint])
      resource, dtypes, shapes, _ = core_ops.prometheus_indexable_init(
          query, metadata=metadata,
          container=scope, shared_name="%s/%s" % (query, uuid.uuid4().hex))
      super(PrometheusValueIOTensor, self).__init__(
          shapes, dtypes, resource, core_ops.prometheus_indexable_get_item,
          internal=internal)
