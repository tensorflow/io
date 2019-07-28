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
"""HDF5 Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_io.core.python.ops import core_ops

def list_hdf5_datasets(filename, **kwargs):
  """list_hdf5_datasets"""
  if not tf.executing_eagerly():
    raise NotImplementedError("list_hdf5_datasets only support eager mode")
  memory = kwargs.get("memory", "")
  datasets, dtypes, shapes = core_ops.list_hdf5_datasets(
      filename, memory=memory)
  entries = zip(tf.unstack(datasets), tf.unstack(dtypes), tf.unstack(shapes))
  entries = [
      (dataset, dtype, tf.boolean_mask(
          shape, tf.math.greater_equal(shape, 0))) for (
              dataset, dtype, shape) in entries]
  return dict([(dataset.numpy().decode(), tf.TensorSpec(
      shape.numpy(), dtype.numpy().decode(), dataset.numpy().decode())) for (
          dataset, dtype, shape) in entries])

def read_hdf5(filename, dataset, start=0, **kwargs):
  """read_hdf5"""
  memory = kwargs.get("memory", "")
  return core_ops.read_hdf5(
      filename,
      dataset.name,
      start=start,
      count=tf.convert_to_tensor(dataset.shape, tf.int64) - start,
      dtype=dataset.dtype,
      memory=memory)

class HDF5Dataset(tf.compat.v1.data.Dataset):
  """A HDF5 Dataset that reads the hdf5 file."""

  def __init__(self, filenames, columns, dtypes=None, shapes=None, batch=None):
    """Create a `HDF5Dataset`.

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      columns: A 0-D or 1-D `tf.int32` tensor containing the columns to extract.
      dtypes: A tuple of `tf.DType` objects representing the types of the
        columns returned.
    """
    self._data_input = core_ops.hdf5_input(
        filenames, ["none", "gz"], columns=columns)
    self._columns = columns
    self._dtypes = dtypes
    self._shapes = shapes
    self._batch = 0 if batch is None else batch
    super(HDF5Dataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return core_ops.hdf5_dataset(
        self._data_input,
        self._batch,
        output_types=self.output_types,
        output_shapes=self.output_shapes)

  @property
  def output_classes(self):
    return tuple([tf.Tensor for _ in self._columns])

  @property
  def output_shapes(self):
    return tuple([tf.TensorShape([]) for _ in self._shapes])

  @property
  def output_types(self):
    return tuple([dtype for dtype in self._dtypes])
