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
"""Avro Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.ops import data_ops

def list_avro_columns(filename, schema, **kwargs):
  """list_avro_columns"""
  if not tf.executing_eagerly():
    raise NotImplementedError("list_avro_columns only support eager mode")
  memory = kwargs.get("memory", "")
  columns, dtypes = core_ops.list_avro_columns(
      filename, schema=schema, memory=memory)
  entries = zip(tf.unstack(columns), tf.unstack(dtypes))
  return dict([(column.numpy().decode(), tf.TensorSpec(
      tf.TensorShape([None]),
      dtype.numpy().decode(),
      column.numpy().decode())) for (
          column, dtype) in entries])

def read_avro(filename, schema, column, **kwargs):
  """read_avro"""
  memory = kwargs.get("memory", "")
  offset = kwargs.get("offset", 0)
  length = kwargs.get("length", -1)
  return core_ops.read_avro(
      filename, schema, column.name, memory=memory,
      offset=offset, length=length, dtype=column.dtype)

class AvroDataset(data_ops.BaseDataset):
  """A Avro Dataset that reads the avro file."""

  def __init__(self, filename, schema, column, **kwargs):
    """Create a `AvroDataset`.

    Args:
      filenames: A string containing one or more filename.
      schema: A string containing the avro schema.
      column: A string containing the column to extract.
    """
    if not tf.executing_eagerly():
      dtype = kwargs.get("dtype")
    else:
      columns = list_avro_columns(filename, schema)
      dtype = columns[column].dtype
    shape = tf.TensorShape([None])

    filesize = tf.io.gfile.GFile(filename).size()
    # capacity is the rough length for each split
    capacity = kwargs.get("capacity", 65536)
    entry_offset = list(range(0, filesize, capacity))
    entry_length = [min(capacity, filesize - offset) for offset in entry_offset]
    dataset = data_ops.BaseDataset.from_tensor_slices(
        (
            tf.constant(entry_offset, tf.int64),
            tf.constant(entry_length, tf.int64)
        )
    ).map(lambda offset, length: core_ops.read_avro(
        filename, schema, column, memory="",
        offset=offset, length=length, dtype=dtype))
    self._dataset = dataset

    super(AvroDataset, self).__init__(
        self._dataset._variant_tensor, [dtype], [shape]) # pylint: disable=protected-access
