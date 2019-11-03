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
"""HDF5Dataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.ops import io_dataset_ops

class _HDF5IODatasetFunction(object):
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

class HDF5IODataset(io_dataset_ops._IODataset): # pylint: disable=protected-access
  """HDF5IODataset"""

  def __init__(self,
               filename,
               dataset,
               internal=True):
    """HDF5IODataset."""
    with tf.name_scope("HDF5IODataset") as scope:
      resource, _ = core_ops.io_hdf5_readable_init(
          filename,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      shape, dtype = core_ops.io_hdf5_readable_spec(resource, dataset)
      shape = tf.TensorShape([None if e < 0 else e for e in shape.numpy()])
      dtype = tf.as_dtype(dtype.numpy())
      capacity = 4096
      super(HDF5IODataset, self).__init__(
          _HDF5IODatasetFunction(
              core_ops.io_hdf5_readable_read,
              resource, dataset, shape, dtype),
          capacity=capacity, internal=internal)
