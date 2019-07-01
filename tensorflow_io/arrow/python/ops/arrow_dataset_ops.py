# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Arrow Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import io
from itertools import chain
import os
import socket
import threading
import tempfile

import tensorflow as tf
from tensorflow import dtypes
from tensorflow.compat.v2 import data
from tensorflow.python.data.ops.dataset_ops import flat_structure
from tensorflow.python.data.util import structure as structure_lib
from tensorflow_io import _load_library
arrow_ops = _load_library('_arrow_ops.so')

if hasattr(tf, "nest"):
  from tensorflow import nest # pylint: disable=ungrouped-imports
else:
  from tensorflow.python.data.util import nest # pylint: disable=ungrouped-imports


def arrow_to_tensor_type(pa_t):
  """Convert Arrow type to tuple of (Tensor dtype, shape dims).
  This function requires pyarrow to be installed.
  """
  import pyarrow as pa
  shape_dims = []  # initialize shape as scalar
  if pa.types.is_boolean(pa_t):
    tf_t = dtypes.bool
  elif pa.types.is_int8(pa_t):
    tf_t = dtypes.int8
  elif pa.types.is_int16(pa_t):
    tf_t = dtypes.int16
  elif pa.types.is_int32(pa_t):
    tf_t = dtypes.int32
  elif pa.types.is_int64(pa_t):
    tf_t = dtypes.int64
  elif pa.types.is_uint8(pa_t):
    tf_t = dtypes.uint8
  elif pa.types.is_uint16(pa_t):
    tf_t = dtypes.uint16
  elif pa.types.is_uint32(pa_t):
    tf_t = dtypes.uint32
  elif pa.types.is_uint64(pa_t):
    tf_t = dtypes.uint64
  elif pa.types.is_float16(pa_t):
    tf_t = dtypes.float16
  elif pa.types.is_float32(pa_t):
    tf_t = dtypes.float32
  elif pa.types.is_float64(pa_t):
    tf_t = dtypes.float64
  elif pa.types.is_list(pa_t):
    if pa.types.is_list(pa_t.value_type):
      raise TypeError("Nested arrays are not currently supported: " + str(pa_t))
    tf_t, shape_dims = arrow_to_tensor_type(pa_t.value_type)
    shape_dims.append(None)  # pyarrow scalar arrays can be variable length
  else:
    raise TypeError("Unsupported type in conversion from Arrow: " + str(pa_t))
  return tf_t, shape_dims


def arrow_schema_to_tensor_types(schema):
  """Convert an Arrow schema to tuple of (Tensor dtypes, TensorShapes).
  This function requires pyarrow to be installed.
  """
  type_shape_list = [arrow_to_tensor_type(field.type) for field in schema]
  tensor_types, shape_dims = zip(*type_shape_list)
  tensor_shapes = tuple(tf.TensorShape(s) for s in shape_dims)
  return tensor_types, tensor_shapes


class ArrowBaseDataset(data.Dataset):
  """Base class for Arrow Datasets to provide columns used in record batches
  and corresponding output tensor types, shapes and classes.
  """

  batch_modes_supported = ('keep_remainder', 'drop_remainder', 'auto')

  def __init__(self,
               make_variant_fn,
               columns,
               output_types,
               output_shapes=None,
               batch_size=None,
               batch_mode='keep_remainder'):
    self._columns = columns
    self._structure = structure_lib.convert_legacy_structure(
        output_types,
        output_shapes or nest.map_structure(
            lambda _: tf.TensorShape(None), output_types),
        nest.map_structure(lambda _: tf.Tensor, output_types))
    self._batch_size = tf.convert_to_tensor(
        batch_size or 0,
        dtype=dtypes.int64,
        name="batch_size")
    if batch_mode not in self.batch_modes_supported:
      raise ValueError(
          "Unsupported batch_mode: '{}', must be one of {}"
          .format(batch_mode, self.batch_modes_supported))
    self._batch_mode = tf.convert_to_tensor(
        batch_mode,
        dtypes.string,
        name="batch_mode")
    if batch_size is not None or batch_mode == 'auto':
      # pylint: disable=protected-access
      self._structure = self._structure._batch(
          batch_size if batch_mode == 'drop_remainder' else None)
    variant_tensor = make_variant_fn(
        columns=self._columns,
        batch_size=self._batch_size,
        batch_mode=self._batch_mode,
        **flat_structure(self))
    super(ArrowBaseDataset, self).__init__(variant_tensor)

  def _inputs(self):
    return []

  @property
  def _element_structure(self):
    return self._structure

  @property
  def columns(self):
    return self._columns

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def batch_mode(self):
    return self._batch_mode


class ArrowDataset(ArrowBaseDataset):
  """An Arrow Dataset from record batches in memory, or a Pandas DataFrame.
  """

  def __init__(self,
               serialized_batches,
               columns,
               output_types,
               output_shapes=None,
               batch_size=None,
               batch_mode='keep_remainder'):
    """Create an ArrowDataset from a Tensor of serialized batches.
    This constructor requires pyarrow to be installed.

    Args:
      serialized_batches: A string Tensor as a serialized buffer containing
                          Arrow record batches as Arrow file format
      columns: A list of column indices to be used in the Dataset
      output_types: Tensor dtypes of the output tensors
      output_shapes: TensorShapes of the output tensors or None to
                     infer partial
      batch_size: Batch size of output tensors, setting a batch size here
                  will create batched Tensors from Arrow memory and can be more
                  efficient than using tf.data.Dataset.batch().
                  NOTE: batch_size does not need to be set if batch_mode='auto'
      batch_mode: Mode of batching, supported strings:
                  "keep_remainder" (default, keeps partial batch data),
                  "drop_remainder" (discard partial batch data),
                  "auto" (size to number of records in Arrow record batch)
    """
    super(ArrowDataset, self).__init__(
        partial(arrow_ops.arrow_dataset, serialized_batches),
        columns,
        output_types,
        output_shapes,
        batch_size,
        batch_mode)

  @classmethod
  def from_record_batches(cls,
                          record_batches,
                          columns,
                          output_types,
                          output_shapes=None,
                          batch_size=None,
                          batch_mode='keep_remainder'):
    """Create an ArrowDataset directly from Arrow record batches.
    This constructor requires pyarrow to be installed.

    Args:
      record_batches: An Arrow record batch or sequence of record batches
      columns: A list of column indices to be used in the Dataset
      output_types: Tensor dtypes of the output tensors
      output_shapes: TensorShapes of the output tensors or None to
                     infer partial
      batch_size: Batch size of output tensors, setting a batch size here
                  will create batched tensors from Arrow memory and can be more
                  efficient than using tf.data.Dataset.batch().
                  NOTE: batch_size does not need to be set if batch_mode='auto'
      batch_mode: Mode of batching, supported strings:
                  "keep_remainder" (default, keeps partial batch data),
                  "drop_remainder" (discard partial batch data),
                  "auto" (size to number of records in Arrow record batch)
    """
    import pyarrow as pa
    if isinstance(record_batches, pa.RecordBatch):
      record_batches = [record_batches]
    if columns is None:
      columns = tuple(range(record_batches[0].num_columns))
    assert record_batches
    buf = io.BytesIO()
    writer = pa.RecordBatchFileWriter(buf, record_batches[0].schema)
    for batch in record_batches:
      writer.write_batch(batch)
    writer.close()
    serialized_batches = tf.convert_to_tensor(
        buf.getvalue(),
        dtype=dtypes.string,
        name="serialized_batches")
    return cls(
        serialized_batches,
        columns,
        output_types,
        output_shapes,
        batch_size,
        batch_mode)

  @classmethod
  def from_pandas(cls,
                  df,
                  columns=None,
                  preserve_index=True,
                  batch_size=None,
                  batch_mode='keep_remainder'):
    """Create an ArrowDataset from a given Pandas DataFrame. Output types
    and shapes are inferred from the Arrow schema after DataFrame conversion.
    If preserve_index is True, the DataFrame index will be the last column.
    This method requires pyarrow to be installed.

    Args:
      df: a Pandas DataFrame
      columns: Optional column indices to use, if None all are used
      preserve_index: Flag to include the DataFrame index as the last column
      batch_size: Batch size of output tensors, setting a batch size here
                  will create batched tensors from Arrow memory and can be more
                  efficient than using tf.data.Dataset.batch().
                  NOTE: batch_size does not need to be set if batch_mode='auto'
      batch_mode: Mode of batching, supported strings:
                  "keep_remainder" (default, keeps partial batch data),
                  "drop_remainder" (discard partial batch data),
                  "auto" (size to number of records in Arrow record batch)
    """
    import pyarrow as pa
    if columns is not None:
      df = df.iloc[:, list(columns)]
    batch = pa.RecordBatch.from_pandas(df, preserve_index=preserve_index)
    columns = tuple(range(batch.num_columns))
    output_types, output_shapes = arrow_schema_to_tensor_types(batch.schema)
    return cls.from_record_batches(
        batch,
        columns,
        output_types,
        output_shapes,
        batch_size,
        batch_mode)


class ArrowFeatherDataset(ArrowBaseDataset):
  """An Arrow Dataset for reading record batches from Arrow feather files.
  Feather is a light-weight columnar format ideal for simple writing of
  Pandas DataFrames. Pyarrow can be used for reading/writing Feather files,
  see https://arrow.apache.org/docs/python/ipc.html#feather-format
  """

  def __init__(self,
               filenames,
               columns,
               output_types,
               output_shapes=None,
               batch_size=None,
               batch_mode='keep_remainder'):
    """Create an ArrowDataset from one or more Feather file names.

    Args:
      filenames: A `tf.string` tensor, Python list or scalar containing files
                 in Arrow Feather format
      columns: A list of column indices to be used in the Dataset
      output_types: Tensor dtypes of the output tensors
      output_shapes: TensorShapes of the output tensors or None to
                     infer partial
      batch_size: Batch size of output tensors, setting a batch size here
                  will create batched tensors from Arrow memory and can be more
                  efficient than using tf.data.Dataset.batch().
                  NOTE: batch_size does not need to be set if batch_mode='auto'
      batch_mode: Mode of batching, supported strings:
                  "keep_remainder" (default, keeps partial batch data),
                  "drop_remainder" (discard partial batch data),
                  "auto" (size to number of records in Arrow record batch)
    """
    filenames = tf.convert_to_tensor(
        filenames,
        dtype=dtypes.string,
        name="filenames")
    super(ArrowFeatherDataset, self).__init__(
        partial(arrow_ops.arrow_feather_dataset, filenames),
        columns,
        output_types,
        output_shapes,
        batch_size,
        batch_mode)

  @classmethod
  def from_schema(cls,
                  filenames,
                  schema,
                  columns=None,
                  batch_size=None,
                  batch_mode='keep_remainder'):
    """Create an Arrow Dataset for reading record batches from Arrow feather
    files, inferring output types and shapes from the given Arrow schema.
    This method requires pyarrow to be installed.

    Args:
      filenames: A `tf.string` tensor, Python list or scalar containing files
                 in Arrow Feather format
      schema: Arrow schema defining the record batch data in the stream
      columns: A list of column indicies to use from the schema, None for all
      batch_size: Batch size of output tensors, setting a batch size here
                  will create batched tensors from Arrow memory and can be more
                  efficient than using tf.data.Dataset.batch().
                  NOTE: batch_size does not need to be set if batch_mode='auto'
      batch_mode: Mode of batching, supported strings:
                  "keep_remainder" (default, keeps partial batch data),
                  "drop_remainder" (discard partial batch data),
                  "auto" (size to number of records in Arrow record batch)
    """
    if columns is None:
      columns = list(range(len(schema)))
    output_types, output_shapes = arrow_schema_to_tensor_types(schema)
    return cls(
        filenames,
        columns,
        output_types,
        output_shapes,
        batch_size,
        batch_mode)


class ArrowStreamDataset(ArrowBaseDataset):
  """An Arrow Dataset for reading record batches from an input stream.
  Currently supported input streams are a socket client or stdin.
  """

  def __init__(self,
               endpoints,
               columns,
               output_types,
               output_shapes=None,
               batch_size=None,
               batch_mode='keep_remainder'):
    """Create an ArrowDataset from an input stream.

    Args:
      endpoints: A `tf.string` tensor, Python list or scalar string defining the
                 input stream.
                 `endpoints` could have the following formats:
                   - "host:port": IPv4 address (default)
                   - "tcp://<host:port>": IPv4 address,
                   - "unix://<path>": local path as unix socket address,
                   - "fd://<number>": STDIN or file descriptor number. For
                     STDIN, use "fd://0" or "fd://-".
      columns: A list of column indices to be used in the Dataset
      output_types: Tensor dtypes of the output tensors
      output_shapes: TensorShapes of the output tensors or None to
                     infer partial
      batch_size: Batch size of output tensors, setting a batch size here
                  will create batched tensors from Arrow memory and can be more
                  efficient than using tf.data.Dataset.batch().
                  NOTE: batch_size does not need to be set if batch_mode='auto'
      batch_mode: Mode of batching, supported strings:
                  "keep_remainder" (default, keeps partial batch data),
                  "drop_remainder" (discard partial batch data),
                  "auto" (size to number of records in Arrow record batch)
    """
    endpoints = tf.convert_to_tensor(
        endpoints,
        dtype=dtypes.string,
        name="endpoints")
    super(ArrowStreamDataset, self).__init__(
        partial(arrow_ops.arrow_stream_dataset, endpoints),
        columns,
        output_types,
        output_shapes,
        batch_size,
        batch_mode)

  @classmethod
  def from_schema(cls,
                  endpoints,
                  schema,
                  columns=None,
                  batch_size=None,
                  batch_mode='keep_remainder'):
    """Create an Arrow Dataset from an input stream, inferring output types
    and shapes from the given Arrow schema.
    This method requires pyarrow to be installed.

    Args:
      endpoints: A `tf.string` tensor, Python list or scalar string defining the
                 input stream.
                 `endpoints` could have the following formats:
                   - "host:port": IPv4 address (default)
                   - "tcp://<host:port>": IPv4 address,
                   - "unix://<path>": local path as unix socket address,
                   - "fd://<number>": STDIN or file descriptor number. For
                     STDIN, use "fd://0" or "fd://-".
      schema: Arrow schema defining the record batch data in the stream
      columns: A list of column indicies to use from the schema, None for all
      batch_size: Batch size of output tensors, setting a batch size here
                  will create batched tensors from Arrow memory and can be more
                  efficient than using tf.data.Dataset.batch().
                  NOTE: batch_size does not need to be set if batch_mode='auto'
      batch_mode: Mode of batching, supported strings:
                  "keep_remainder" (default, keeps partial batch data),
                  "drop_remainder" (discard partial batch data),
                  "auto" (size to number of records in Arrow record batch)
    """
    if columns is None:
      columns = list(range(len(schema)))
    output_types, output_shapes = arrow_schema_to_tensor_types(schema)
    return cls(
        endpoints,
        columns,
        output_types,
        output_shapes,
        batch_size,
        batch_mode)

  @classmethod
  def from_record_batches(cls,
                          record_batch_iter,
                          output_types,
                          output_shapes=None,
                          columns=None,
                          batch_size=None,
                          batch_mode='keep_remainder'):
    """Create an ArrowStreamDataset by serving a sequence of Arrow record
    batches in a background thread.
    This constructor requires pyarrow to be installed.

    Args:
      record_batch_iter: A sequence or iterator of Arrow record batches
      output_types: Tensor dtypes of the output tensors
      output_shapes: TensorShapes of the output tensors or None to
                     infer partial
      columns: Optional list of column indices to be used, if None all are used
      batch_size: Batch size of output tensors, setting a batch size here
                  will create batched tensors from Arrow memory and can be more
                  efficient than using tf.data.Dataset.batch().
                  NOTE: batch_size does not need to be set if batch_mode='auto'
      batch_mode: Mode of batching, supported strings:
                  "keep_remainder" (default, keeps partial batch data),
                  "drop_remainder" (discard partial batch data),
                  "auto" (size to number of records in Arrow record batch)
    """
    import pyarrow as pa

    # Create a UDS server by default if not Windows
    if os.name != "nt":
      sock_path = os.path.join(tempfile.gettempdir(), 'arrow_io_stream.sock')
      endpoint = 'unix://{}'.format(sock_path)
      try:
        os.unlink(sock_path)
      except OSError:
        if os.path.exists(sock_path):
          raise
      sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
      sock.bind(sock_path)
    # Create a TCP server
    else:
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      sock.bind(('127.0.0.1', 0))
      host_addr, port = sock.getsockname()
      endpoint = '{}:{}'.format(host_addr, port)
    sock.listen(1)

    def run_server():
      """serve record batches"""
      conn, _ = sock.accept()
      outfile = conn.makefile(mode='wb')
      writer = None
      for batch in record_batch_iter:
        if writer is None:
          writer = pa.RecordBatchStreamWriter(outfile, batch.schema)
        writer.write_batch(batch)
      writer.close()
      outfile.close()
      conn.close()
      sock.close()

    # Run the server in a thread
    server = threading.Thread(target=run_server)
    server.start()

    if columns is None:
      columns = list(range(len(output_types)))

    return cls(
        endpoint,
        columns,
        output_types,
        output_shapes,
        batch_size,
        batch_mode)

  @classmethod
  def from_pandas(cls,
                  data_frames,
                  columns=None,
                  preserve_index=True,
                  batch_size=None):
    """Create an ArrowStreamDataset by serving a DataFrame, or batches of a
    DataFrame in a background thread.
    This constructor requires pandas and pyarrow to be installed.

    Args:
      df: A Pandas DataFrame or sequence of DataFrames
      columns: Optional column indices to use, if None all are used
      preserve_index: Flag to include the DataFrame index as the last column
      batch_size: Batch size of output tensors, setting a batch size here
                  will create batched tensors from Arrow memory and can be more
                  efficient than using tf.data.Dataset.batch().
                  NOTE: Currently, only 'keep_remainder' batch mode supported
    """
    import pandas as pd
    import pyarrow as pa
    if isinstance(data_frames, pd.DataFrame):
      data_frames = [data_frames]

    def gen_record_batches():
      """record batch generator"""
      for df in data_frames:
        if columns is not None:
          df = df.iloc[:, list(columns)]

        # Compute step size (round int up) if slicing into batches
        step = None if batch_size is None else -(-len(df) // batch_size)
        for start in range(0, 1 if step is None else len(df), step):
          df_slice = df if step is None else df[start:start + step]
          batch = pa.RecordBatch.from_pandas(
              df_slice, preserve_index=preserve_index)
          yield batch

    # Get first batch to convert schema to output types and shapes
    record_batch_iter = gen_record_batches()
    batch = next(record_batch_iter)
    output_types, output_shapes = arrow_schema_to_tensor_types(batch.schema)

    return cls.from_record_batches(
        chain([batch], record_batch_iter),
        output_types,
        output_shapes,
        batch_size=batch_size,
        batch_mode='keep_remainder')
