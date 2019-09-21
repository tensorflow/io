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
"""AvroIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops

class AvroIOTensor(io_tensor_ops._TableIOTensor): # pylint: disable=protected-access
  """AvroIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               schema,
               capacity=None,
               internal=False):
    with tf.name_scope("AvroIOTensor") as scope:
      metadata = ["schema: %s" % schema]
      resource, columns = core_ops.avro_indexable_init(
          filename, metadata=metadata,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      partitions = None
      if capacity is not None:
        partitions = core_ops.avro_indexable_partitions(resource)
        partitions = partitions.numpy().tolist()
        if capacity > 0:
          partitions = [
              v for e in partitions for v in list(
                  [capacity] * (e // capacity) + [e % capacity])]
      columns = [column.decode() for column in columns.numpy().tolist()]
      spec = []
      for column in columns:
        shape, dtype = core_ops.avro_indexable_spec(resource, column)
        shape = tf.TensorShape(shape.numpy())
        dtype = tf.as_dtype(dtype.numpy())
        spec.append(tf.TensorSpec(shape, dtype, column))
      spec = tuple(spec)
      super(AvroIOTensor, self).__init__(
          spec, columns,
          resource, core_ops.avro_indexable_read,
          partitions=partitions, internal=internal)
