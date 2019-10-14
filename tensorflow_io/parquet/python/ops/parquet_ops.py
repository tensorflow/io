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
"""Parquet Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops as parquet_ops
from tensorflow_io.core.python.ops import data_ops

warnings.warn(
    "The tensorflow_io.parquet.ParquetDataset is "
    "deprecated. Please look for tfio.IOTensor.from_parquet "
    "for reading parquet files into tensorflow.",
    DeprecationWarning)

def list_parquet_columns(filename, **kwargs):
  """list_parquet_columns"""
  if not tf.executing_eagerly():
    raise NotImplementedError("list_parquet_columns only support eager mode")
  memory = kwargs.get("memory", "")
  columns, dtypes, shapes = parquet_ops.io_list_parquet_columns(
      filename, memory=memory)
  entries = zip(tf.unstack(columns), tf.unstack(dtypes), tf.unstack(shapes))
  return dict([(column.numpy().decode(), tf.TensorSpec(
      shape.numpy(), dtype.numpy().decode(), column.numpy().decode())) for (
          column, dtype, shape) in entries])

def read_parquet(filename, column, **kwargs):
  """read_parquet"""
  memory = kwargs.get("memory", "")
  start = kwargs.get("start", 0)
  stop = kwargs.get("stop", None)
  if stop is None and column.shape[0] is not None:
    stop = column.shape[0] - start
  if stop is None:
    stop = -1
  return parquet_ops.io_read_parquet(
      filename, column.name, memory=memory,
      start=start, stop=-1, dtype=column.dtype)

class ParquetDataset(data_ops.BaseDataset):
  """A Parquet Dataset that reads the parquet file."""

  def __init__(self, filename, column, **kwargs):
    """Create a `ParquetDataset`.

    `ParquetDataset` allows a user to read data from a parquet file.

    Args:
      filename: filename of the parquet file to read.
      column: column name to read.
    """
    # Note: start, stop and dtype could be in kwargs if in graph mode.
    if not tf.executing_eagerly():
      start = kwargs.get("start")
      stop = kwargs.get("stop")
      dtype = kwargs.get("dtype")
    else:
      columns = list_parquet_columns(filename)
      start = 0
      stop = columns[column].shape[0]
      dtype = columns[column].dtype

    shape = tf.TensorShape([None])

    # capacity is the rough count for each chunk in dataset
    capacity = kwargs.get("capacity", 65536)
    entry_start = list(range(start, stop, capacity))
    entry_stop = entry_start[1:] + [stop]
    dataset = data_ops.BaseDataset.from_tensor_slices(
        (tf.constant(entry_start, tf.int64), tf.constant(entry_stop, tf.int64))
    ).map(lambda start, stop: parquet_ops.io_read_parquet(
        filename, column, memory="", start=start, stop=stop, dtype=dtype))
    self._dataset = dataset

    super(ParquetDataset, self).__init__(
        self._dataset._variant_tensor, [dtype], [shape]) # pylint: disable=protected-access
