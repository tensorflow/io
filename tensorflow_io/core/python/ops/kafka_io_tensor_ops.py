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
      resource, shapes, dtypes = core_ops.kafka_indexable_init(
          subscription, metadata=metadata,
          container=scope,
          shared_name="%s/%s" % (subscription, uuid.uuid4().hex))
      shapes = [
          tf.TensorShape(
              [None if dim < 0 else dim for dim in e.numpy() if dim != 0]
          ) for e in tf.unstack(shapes)]
      dtypes = [tf.as_dtype(e.numpy()) for e in tf.unstack(dtypes)]
      assert len(shapes) == 1
      assert len(dtypes) == 1
      shape = shapes[0]
      dtype = dtypes[0]
      spec = tf.TensorSpec(shape, dtype)

      super(KafkaIOTensor, self).__init__(
          spec, resource, core_ops.kafka_indexable_get_item,
          internal=internal)
