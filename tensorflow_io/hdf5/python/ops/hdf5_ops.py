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
from tensorflow.compat.v1 import data
from tensorflow_io.core.python.ops import core_ops as hdf5_ops

class HDF5Dataset(data.Dataset):
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
    self._data_input = hdf5_ops.hdf5_input(
        filenames, ["none", "gz"], columns=columns)
    self._columns = columns
    self._dtypes = dtypes
    self._shapes = shapes
    self._batch = 0 if batch is None else batch
    super(HDF5Dataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return hdf5_ops.hdf5_dataset(
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
