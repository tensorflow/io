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

import warnings

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.ops import data_ops

warnings.warn(
    "The tensorflow_io.hdf5.HDF5Dataset is "
    "deprecated. Please look for tfio.IOTensor.from_hdf5 "
    "for reading HDF5 files into tensorflow.",
    DeprecationWarning)

def list_hdf5_datasets(filename, **kwargs):
  """list_hdf5_datasets"""
  if not tf.executing_eagerly():
    raise NotImplementedError("list_hdf5_datasets only support eager mode")
  memory = kwargs.get("memory", "")
  datasets, dtypes, shapes = core_ops.io_list_hdf5_datasets(
      filename, memory=memory)
  entries = zip(tf.unstack(datasets), tf.unstack(dtypes), tf.unstack(shapes))
  entries = [
      (dataset, dtype, tf.boolean_mask(
          shape, tf.math.greater_equal(shape, 0))) for (
              dataset, dtype, shape) in entries]
  return dict([(dataset.numpy().decode(), tf.TensorSpec(
      shape.numpy(), dtype.numpy().decode(), dataset.numpy().decode())) for (
          dataset, dtype, shape) in entries])

def read_hdf5(filename, dataset, **kwargs):
  """read_hdf5"""
  memory = kwargs.get("memory", "")
  start = kwargs.get("start", 0)
  stop = kwargs.get("stop", None)
  if stop is None and dataset.shape[0] is not None:
    stop = dataset.shape[0] - start
  if stop is None:
    stop = -1
  return core_ops.io_read_hdf5(
      filename, dataset.name, memory=memory,
      start=start, stop=stop, dtype=dataset.dtype)

class HDF5Dataset(data_ops.BaseDataset):
  """A HDF5 Dataset that reads the hdf5 file."""

  def __init__(self, filename, dataset, **kwargs):
    """Create a `HDF5Dataset`.

    Args:
      filename: A string of th hdf5 filename.
      dataset: A string of the dataset name.
    """
    if not tf.executing_eagerly():
      start = kwargs.get("start")
      stop = kwargs.get("stop")
      dtype = kwargs.get("dtype")
      shape = kwargs.get("shape")
    else:
      datasets = list_hdf5_datasets(filename)
      start = 0
      stop = datasets[dataset].shape[0]
      dtype = datasets[dataset].dtype
      shape = tf.TensorShape(
          [None if i == 0 else e for i, e in enumerate(
              datasets[dataset].shape.as_list())])

    # capacity is the rough count for each chunk in dataset
    capacity = kwargs.get("capacity", 65536)
    entry_start = list(range(start, stop, capacity))
    entry_stop = entry_start[1:] + [stop]
    self._dataset = data_ops.BaseDataset.from_tensor_slices(
        (tf.constant(entry_start, tf.int64), tf.constant(entry_stop, tf.int64))
    ).map(lambda start, stop: core_ops.io_read_hdf5(
        filename, dataset, memory="", start=start, stop=stop, dtype=dtype))
    super(HDF5Dataset, self).__init__(
        self._dataset._variant_tensor, [dtype], [shape]) # pylint: disable=protected-access
