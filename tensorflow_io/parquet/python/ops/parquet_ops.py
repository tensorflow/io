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

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops as parquet_ops
from tensorflow_io.core.python.ops import data_ops

def read_parquet_specs(filename):
  """read_parquet_specs"""
  if not tf.executing_eagerly():
    raise NotImplementedError("read_parquet_spect only support eager mode")
  columns, dtypes, shapes = parquet_ops.read_parquet_specs(filename)
  entries = zip(tf.unstack(columns), tf.unstack(dtypes), tf.unstack(shapes))
  return dict([(column.numpy(), tf.TensorSpec(
      shape.numpy(), dtype.numpy(), column.numpy())) for (
          column, dtype, shape) in entries])

def read_parquet(filename, spec, start=0, **kwargs):
  """read_parquet"""
  memory = kwargs.get("memory", "")
  return parquet_ops.read_parquet(
      filename, spec.name,
      start=start, count=spec.shape[0] - start, dtype=spec.dtype,
      memory=memory)

class ParquetDataset(data_ops.BaseDataset):
  """A Parquet Dataset that reads the parquet file."""

  def __init__(self, filename, column, batch=None, **kwargs):
    """Create a `ParquetDataset`.

    `ParquetDataset` allows a user to read data from a parquet file.

    Args:
      filename: filename of the parquet file to read.
      column: column name to read.
    """
    # Note: count and dtype could be in kwargs if in graph mode.
    if not tf.executing_eagerly():
      count = kwargs.get("count")
      dtype = kwargs.get("dtype")
    else:
      specs = read_parquet_specs(filename)
      count = specs[column].shape[0]
      dtype = specs[column].dtype

    batch = 0 if batch is None else batch
    shape = tf.TensorShape([]) if (
        batch is None or batch == 0) else tf.TensorShape([None])

    # capacity is the rough count for each chunk in dataset
    # not directly related to batch, will be padded to batch though
    capacity = kwargs.get("capacity", 65536)
    if batch is not None and batch != 0 and capacity > batch:
      capacity = (capacity // batch) * batch
    entry_start = range(0, count, capacity)
    entry_count = [min(capacity, count - start) for start in entry_start]
    dataset = data_ops.BaseDataset.from_tensor_slices(
        (tf.constant(entry_start, tf.int64), tf.constant(entry_count, tf.int64))
    ).map(lambda start, count: parquet_ops.read_parquet(
        filename, column, start, count, dtype=dtype, memory=""))
    if batch is None or batch == 0:
      self._dataset = dataset.unbatch()
    else:
      # TODO: convert to rebatch for performance
      self._dataset = dataset.unbatch().batch(batch)

    super(ParquetDataset, self).__init__(
        self._dataset._variant_tensor, [dtype], [shape]) # pylint: disable=protected-access
