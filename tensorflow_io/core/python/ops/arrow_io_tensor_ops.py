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

class _ArrowIOTensorComponentFunction(io_tensor_ops._IOTensorComponentFunction): # pylint: disable=protected-access
  """_ArrowIOTensorComponentFunction will translate call"""
  def __init__(self,
               function,
               resource,
               component, column_index, shape, dtype):
    super(_ArrowIOTensorComponentFunction, self).__init__(
        function, resource, component, shape, dtype)
    self._column_index = column_index
  def __call__(self, start, stop):
    start, stop, _ = slice(start, stop).indices(self._length)
    return self._function(
        self._resource,
        start=start, stop=stop,
        column_index=self._column_index,
        shape=self._shape, dtype=self._dtype)


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


class ArrowIOTensor(io_tensor_ops._TableIOTensor): # pylint: disable=protected-access
  """ArrowIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               table,
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
            resource, column, column_index, shape, dtype)
        elements.append(
            io_tensor_ops.BaseIOTensor(
                spec, function, internal=internal))

      spec = tuple([e.spec for e in elements])
      super(ArrowIOTensor, self).__init__(
          spec, columns, elements, internal=internal)
