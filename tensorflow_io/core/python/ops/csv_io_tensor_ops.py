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
"""CSVIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops

class CSVIOTensor(io_tensor_ops._TableIOTensor): # pylint: disable=protected-access
  """CSVIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               internal=False):
    with tf.name_scope("CSVIOTensor") as scope:
      resource, columns = core_ops.csv_indexable_init(
          filename,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      columns = [column.decode() for column in columns.numpy().tolist()]
      spec = []
      for column in columns:
        shape, dtype = core_ops.csv_indexable_spec(resource, column)
        shape = tf.TensorShape(shape.numpy())
        dtype = tf.as_dtype(dtype.numpy())
        spec.append(tf.TensorSpec(shape, dtype, column))
      spec = tuple(spec)
      super(CSVIOTensor, self).__init__(
          spec, columns,
          resource, core_ops.csv_indexable_read,
          partitions=None,
          internal=internal)

  #=============================================================================
  # IsNull checking
  #=============================================================================
  def isnull(self, column):
    """Return a BaseIOTensor of bool for null values in `column`"""
    column_index = self.columns.index(
        next(e for e in self.columns if e == column))
    spec = tf.nest.flatten(self.spec)[column_index]
    # change spec to bool
    spec = tf.TensorSpec(spec.shape, tf.bool)
    class _Function(object):
      def __init__(self, func, spec, column):
        self._func = func
        self._shape = tf.TensorShape([None]).concatenate(spec.shape[1:])
        self._dtype = spec.dtype
        self._component = column

      def __call__(self, resource, start, stop):
        return self._func(
            resource, start=start, stop=stop,
            component=self._component, filter=['label'],
            shape=self._shape, dtype=self._dtype)

    return io_tensor_ops.BaseIOTensor(
        spec, self._resource,
        _Function(core_ops.csv_indexable_read, spec, column),
        partitions=None, internal=True)
