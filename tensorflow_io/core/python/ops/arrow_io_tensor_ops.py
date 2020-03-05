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
"""FeatherIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops


def _extract_table_arrays(table):
  """Get buffer info from arrays in table, outputs are padded so dim sizes
     are rectangular.

     Args:
       table: A pyarrow.Table
     Return:
       tuple of:
         array_buffer_addrs: 3-dim list of buffer addresses where dims are
                             columns, chunks, buffer addresses
         array_buffer_sizes: 3-dim list of buffer sizes, follows addrs layout
         array_lengths: 3-dim list of array lengths where dims are columns,
                        chunks, length of array followed by child array lengths
  """
  array_buffer_addrs = []
  array_buffer_sizes = []
  array_lengths = []
  max_num_bufs = 0
  max_num_chunks = 0
  max_num_lengths = 0

  # Iterate over each column in the Table
  for chunked_array in table:
    array_chunk_buffer_addrs = []
    array_chunk_buffer_sizes = []
    array_chunk_lengths = []

    # Iterate over each data chunk in the column
    for arr in chunked_array.iterchunks():
      bufs = arr.buffers()
      array_chunk_buffer_addrs.append(
          [b.address if b is not None else 0 for b in bufs])
      array_chunk_buffer_sizes.append(
          [b.size if b is not None else 0 for b in bufs])

      # Get the total length of the array followed by lenghts of children
      array_and_child_lengths = [len(arr)]

      # Check if has child array, e.g. list type
      if arr.type.num_children > 0:
        if hasattr(arr, 'values'):
          array_and_child_lengths.append(len(arr.values))
        else:
          raise ValueError("Only nested type currently supported is ListType")

      array_chunk_lengths.append(array_and_child_lengths)
      if len(bufs) > max_num_bufs:
        max_num_bufs = len(bufs)
      if len(array_and_child_lengths) > max_num_lengths:
        max_num_lengths = len(array_and_child_lengths)

    array_buffer_addrs.append(array_chunk_buffer_addrs)
    array_buffer_sizes.append(array_chunk_buffer_sizes)
    array_lengths.append(array_chunk_lengths)
    if len(array_chunk_lengths) > max_num_chunks:
      max_num_chunks = len(array_chunk_lengths)

  # Pad buffer addrs, sizes and array lengths so inputs are rectangular
  num_columns = len(array_buffer_sizes)
  for i in range(num_columns):

    # pad chunk list with empty lists that will be padded with null bufs
    if len(array_buffer_sizes[i]) < max_num_chunks:
      array_buffer_sizes[i].extend([[]] * (max_num_chunks -
                                           len(array_buffer_sizes[i])))
    if len(array_lengths[i]) < max_num_chunks:
      array_lengths[i].extend([-1] * (max_num_chunks - len(array_lengths[i])))

    num_chunks = len(array_buffer_sizes[i])
    for j in range(num_chunks):

      # pad buffer addr, size, and array length lists
      if len(array_buffer_sizes[i][j]) < max_num_bufs:
        array_buffer_sizes[i][j].extend([-1] * (max_num_bufs -
                                                len(array_buffer_sizes[i][j])))
        array_buffer_addrs[i][j].extend([0] * (max_num_bufs -
                                               len(array_buffer_addrs[i][j])))
      if len(array_lengths[i][j]) < max_num_lengths:
        array_lengths[i][j].extend([-1] * (max_num_lengths -
                                           len(array_lengths[i][j])))
  return array_buffer_addrs, array_buffer_sizes, array_lengths


class _ArrowIOTensorComponentFunction():
  """_ArrowIOTensorComponentFunction will translate call"""
  def __init__(self,
               function,
               resource,
               column_index,
               shape,
               dtype):
    super(_ArrowIOTensorComponentFunction, self).__init__()
    self._function = function
    self._resource = resource
    self._column_index = column_index
    self._shape = shape
    self._dtype = dtype
  def __call__(self, start, stop):
    # get the start and stop, and use 0 (start) and -1 (stop) if needed
    start = start or 0
    stop = stop or -1
    return self._function(
        self._resource,
        self._column_index,
        self._shape,
        start, stop,
        dtype=self._dtype)
  @property
  def length(self):
    return self._shape[0]


class ArrowBaseIOTensor(io_tensor_ops.BaseIOTensor):
  """ArrowBaseIOTensor"""
  def __init__(self, shape, dtype, spec, function, internal=False):
    super(ArrowBaseIOTensor, self).__init__(spec, function, internal=internal)
    self._shape = shape
    self._dtype = dtype
    self._spec = spec

  @property
  def spec(self):
    """Returns the TensorSpec of the tensor"""
    return self._spec

  @property
  def shape(self):
    """Returns the `TensorShape` that represents the shape of the tensor."""
    return self._shape

  @property
  def dtype(self):
    """Returns the `dtype` of elements in the tensor."""
    return self._dtype


class ArrowIOTensor(io_tensor_ops._TableIOTensor): # pylint: disable=protected-access
  """ArrowIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               table,
               spec=None,
               internal=False):
    with tf.name_scope("ArrowIOTensor") as scope:

      # Hold reference to table and schema buffer for life of this op
      self._table = table
      self._schema_buffer = table.schema.serialize()

      # Get buffer addresses as long ints
      schema_buffer_addr = self._schema_buffer.address
      schema_buffer_size = self._schema_buffer.size
      array_buffer_addrs, array_buffer_sizes, array_lengths = \
          _extract_table_arrays(table)

      # Create the Arrow readable resource
      resource = core_ops.io_arrow_readable_from_memory_init(
          schema_buffer_addr,
          schema_buffer_size,
          array_buffer_addrs,
          array_buffer_sizes,
          array_lengths,
          container=scope,
          shared_name="pyarrow.Table%s/%s" % (
              table.schema.names, uuid.uuid4().hex))

      if tf.executing_eagerly():
        # Create a BaseIOTensor for each column
        elements = []
        columns = table.column_names
        for column_index, column in enumerate(columns):
          shape, dtype = core_ops.io_arrow_readable_spec(resource, column_index)
          shape = tf.TensorShape(shape.numpy())
          dtype = tf.as_dtype(dtype.numpy())
          spec = tf.TensorSpec(shape, dtype, column)
          function = _ArrowIOTensorComponentFunction( # pylint: disable=protected-access
              core_ops.io_arrow_readable_read,
              resource, column_index, shape, dtype)
          elements.append(
              ArrowBaseIOTensor(
                  shape, dtype, spec, function, internal=internal))
        spec = tuple([e.spec for e in elements])
      else:
        assert spec is not None
        columns, entries = zip(*spec.items())
        dtypes = [
            entry if isinstance(
                entry, tf.dtypes.DType) else entry.dtype for entry in entries]

        col_to_idx = {column: i for i, column in enumerate(table.column_names)}

        shapes = []
        for col in columns:
          shape, _ = core_ops.io_arrow_readable_spec(resource, col_to_idx[col])
          shapes.append(shape)

        entries = [
            tf.TensorSpec(None, dtype, column) for (
                dtype, column) in zip(dtypes, columns)]

        elements = []
        for (entry, shape) in zip(entries, shapes):
          function = _ArrowIOTensorComponentFunction(
              core_ops.io_arrow_readable_read,
              resource, col_to_idx[entry.name], shape, entry.dtype)
          elements.append(
              ArrowBaseIOTensor(
                  shape, entry.dtype, entry, function, internal=internal))
        spec = tuple(entries)

      super(ArrowIOTensor, self).__init__(
          spec, columns, elements, internal=internal)
