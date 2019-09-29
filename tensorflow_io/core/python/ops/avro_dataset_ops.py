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
"""AvroDataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.ops import io_dataset_ops

class _AvroIODatasetFunction(object):
  def __init__(self, function, resource, component, shape, dtype):
    self._function = function
    self._resource = resource
    self._component = component
    self._shape = tf.TensorShape([None]).concatenate(shape[1:])
    self._dtype = dtype
  def __call__(self, start, stop):
    return self._function(
        self._resource, start=start, stop=stop,
        component=self._component, shape=self._shape, dtype=self._dtype)

class AvroIODataset(io_dataset_ops._IODataset): # pylint: disable=protected-access
  """AvroIODataset"""

  def __init__(self,
               filename,
               schema,
               column,
               internal=True):
    """AvroIODataset."""
    with tf.name_scope("AvroIODataset") as scope:
      metadata = ["schema: %s" % schema]
      resource, _ = core_ops.avro_readable_init(
          filename, metadata=metadata,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      shape, dtype = core_ops.avro_readable_spec(resource, column)
      shape = tf.TensorShape([None if e < 0 else e for e in shape.numpy()])
      dtype = tf.as_dtype(dtype.numpy())
      capacity = 4096
      super(AvroIODataset, self).__init__(
          _AvroIODatasetFunction(
              core_ops.avro_readable_read,
              resource, column, shape, dtype), capacity=capacity,
          internal=internal)
