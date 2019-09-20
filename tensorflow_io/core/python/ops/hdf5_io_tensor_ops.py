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
"""HDF5IOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops

class HDF5IOTensor(io_tensor_ops._CollectionIOTensor): # pylint: disable=protected-access
  """HDF5IOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               internal=False):
    with tf.name_scope("HDF5IOTensor") as scope:
      class _Function(object):
        def __init__(self, func, spec, key):
          self._func = func
          self._shape = tf.TensorShape([None]).concatenate(spec.shape[1:])
          self._dtype = spec.dtype
          self._component = key
        def __call__(self, resource, start, stop):
          return self._func(
              resource, start=start, stop=stop,
              component=self._component,
              shape=self._shape, dtype=self._dtype)

      resource, columns = core_ops.hdf5_indexable_init(
          filename,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      columns = [column.decode() for column in columns.numpy().tolist()]
      elements = []
      for column in columns:
        shape, dtype = core_ops.hdf5_indexable_spec(resource, column)
        shape = tf.TensorShape(shape)
        dtype = tf.as_dtype(dtype.numpy())
        spec = tf.TensorSpec(shape, dtype, column)
        elements.append(
            io_tensor_ops.BaseIOTensor(
                spec, resource,
                _Function(core_ops.hdf5_indexable_read, spec, column),
                partitions=None, internal=True))
      spec = tuple([e.spec for e in elements])
      super(HDF5IOTensor, self).__init__(
          spec, columns, elements,
          internal=internal)
