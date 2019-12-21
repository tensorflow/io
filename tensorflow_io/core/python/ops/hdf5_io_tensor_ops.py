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
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.ops import io_tensor_ops

class _HDF5IOTensorFunction(object):
  """_HDF5IOTensorFunction will translate call"""
  def __init__(self, function, resource, component, shape, dtype):
    self._function = function
    self._resource = resource
    self._component = component
    self._length = shape[0]
    self._shape = tf.TensorShape([None]).concatenate(shape[1:])
    self._dtype = dtype
  def __call__(self, start, stop):
    start, stop, _ = slice(start, stop).indices(self._length)
    shape = tf.TensorShape([stop - start]).concatenate(self._shape[1:])
    return self._function(
        self._resource, start=start, shape=shape,
        component=self._component, dtype=self._dtype)
  @property
  def length(self):
    return self._length

class HDF5IOTensor(io_tensor_ops._CollectionIOTensor): # pylint: disable=protected-access
  """HDF5IOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               internal=False):
    with tf.name_scope("HDF5IOTensor") as scope:
      # TODO: unique shared_name might be removed if HDF5 is thead-safe?
      resource, columns = core_ops.io_hdf5_readable_init(
          filename,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      columns = [column.decode() for column in columns.numpy().tolist()]
      elements = []
      for column in columns:
        shape, dtype = core_ops.io_hdf5_readable_spec(resource, column)
        shape = tf.TensorShape(shape.numpy())
        dtype = tf.as_dtype(dtype.numpy())
        spec = tf.TensorSpec(shape, dtype, column)
        if shape.rank == 0:
          value = core_ops.io_hdf5_readable_read(
              resource, 0, shape, column, dtype)
          elements.append(
              io_tensor_ops.ScalarIOTensor(
                  spec, value, internal=internal))
        else:
          function = _HDF5IOTensorFunction(
              core_ops.io_hdf5_readable_read,
              resource, column, shape, dtype)
          elements.append(
              io_tensor_ops.BaseIOTensor(
                  spec, function, internal=internal))
      spec = tuple([e.spec for e in elements])
      super(HDF5IOTensor, self).__init__(
          spec, columns, elements,
          internal=internal)
