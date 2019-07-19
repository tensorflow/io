# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for ArrowDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import io
import os
import sys
import socket
import tempfile
import threading
import pytest

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()

from tensorflow import dtypes # pylint: disable=wrong-import-position
from tensorflow import errors # pylint: disable=wrong-import-position
from tensorflow import test   # pylint: disable=wrong-import-position

import tensorflow_io.arrow as arrow_io # pylint: disable=wrong-import-position

if sys.version_info == (3, 4):
  pytest.skip(
      "pyarrow is not supported with python 3.4", allow_module_level=True)

import pyarrow as pa  # pylint: disable=wrong-import-order,wrong-import-position
from pyarrow.feather import write_feather # pylint: disable=wrong-import-order,wrong-import-position


TruthData = namedtuple("TruthData", ["data", "output_types", "output_shapes"])


class ArrowDatasetTest(test.TestCase):
  """ArrowDatasetTest"""
  @classmethod
  def setUpClass(cls): # pylint: disable=invalid-name
    """setUpClass"""
    cls.scalar_data = [
        [True, False, True, True],
        [1, 2, -3, 4],
        [1, 2, -3, 4],
        [1, 2, -3, 4],
        [1, 2, -3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1.1, 2.2, 3.3, 4.4],
        [1.1, 2.2, 3.3, 4.4],
    ]
    cls.scalar_dtypes = (
        dtypes.bool,
        dtypes.int8,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint8,
        dtypes.uint16,
        dtypes.uint32,
        dtypes.uint64,
        dtypes.float32,
        dtypes.float64
    )
    cls.scalar_shapes = tuple(
        [tf.TensorShape([]) for _ in cls.scalar_dtypes])

    cls.list_data = [
        [[1, 1], [2, 2], [3, 3], [4, 4]],
        [[1], [2, 2], [3, 3, 3], [4, 4, 4]],
        [[1, 1], [2, 2], [3, 3], [4, 4]],
        [[1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4]],
        [[1.1], [2.2, 2.2], [3.3, 3.3, 3.3], [4.4, 4.4, 4.4]],
        [[1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4]],
    ]
    cls.list_dtypes = (
        dtypes.int32,
        dtypes.int32,
        dtypes.int64,
        dtypes.float32,
        dtypes.float32,
        dtypes.float64
    )
    cls.list_shapes = tuple(
        [tf.TensorShape([None]) for _ in cls.list_dtypes])

  def run_test_case(self, dataset, truth_data, batch_size=None):
    """run_test_case"""

    def is_float(dtype):
      """Check if dtype is a floating-point"""
      return dtype in [dtypes.float16, dtypes.float32, dtypes.float64]

    def evaluate_result(value):
      """Check the results match truth data"""
      for i, col in enumerate(dataset.columns):
        if truth_data.output_shapes[col].ndims == 0:
          if is_float(truth_data.output_types[col]):
            self.assertAlmostEqual(value[i], truth_data.data[col][row], 4)
          else:
            self.assertEqual(value[i], truth_data.data[col][row])
        elif truth_data.output_shapes[col].ndims == 1:
          if is_float(truth_data.output_types[col]):
            for j, v in enumerate(value[i]):
              self.assertAlmostEqual(v, truth_data.data[col][row][j], 4)
          else:
            self.assertListEqual(value[i].tolist(), truth_data.data[col][row])

    # Row counter for each single result or batch of multiple rows
    row = 0

    # Iterate over the dataset
    for results in dataset:

      # For batches, iterate over each row in batch or remainder at end
      for result_idx in range(batch_size or 1):

        # Get a single row value
        if batch_size is None:
          value = [r.numpy() for r in results]
        # Get a batch of values and check 1 row at a time
        else:
          if result_idx == 0:
            value_batch = [r.numpy() for r in results]

          # Check for a partial result
          if result_idx == value_batch[0].shape[0]:
            break

          # Get a single row out of the batch
          value = [v[result_idx] for v in value_batch]

        # Check the result then increment the row counter
        evaluate_result(value)
        row += 1

    # Check that all data was returned by Dataset
    self.assertEqual(row, len(truth_data.data[0]))

  def get_arrow_type(self, dt, is_list):
    """get_arrow_type"""
    if dt == dtypes.bool:
      arrow_type = pa.bool_()
    elif dt == dtypes.int8:
      arrow_type = pa.int8()
    elif dt == dtypes.int16:
      arrow_type = pa.int16()
    elif dt == dtypes.int32:
      arrow_type = pa.int32()
    elif dt == dtypes.int64:
      arrow_type = pa.int64()
    elif dt == dtypes.uint8:
      arrow_type = pa.uint8()
    elif dt == dtypes.uint16:
      arrow_type = pa.uint16()
    elif dt == dtypes.uint32:
      arrow_type = pa.uint32()
    elif dt == dtypes.uint64:
      arrow_type = pa.uint64()
    elif dt == dtypes.float16:
      arrow_type = pa.float16()
    elif dt == dtypes.float32:
      arrow_type = pa.float32()
    elif dt == dtypes.float64:
      arrow_type = pa.float64()
    else:
      raise TypeError("Unsupported dtype for Arrow" + str(dt))
    if is_list:
      arrow_type = pa.list_(arrow_type)
    return arrow_type

  def make_record_batch(self, truth_data):
    """Make an Arrow RecordBatch for given test data"""
    arrays = [pa.array(truth_data.data[col],
                       type=self.get_arrow_type(
                           truth_data.output_types[col],
                           truth_data.output_shapes[col].ndims == 1))
              for col in range(len(truth_data.output_types))]
    names = ["%s_[%s]" % (i, a.type) for i, a in enumerate(arrays)]
    return pa.RecordBatch.from_arrays(arrays, names)

  def test_arrow_dataset(self):
    """test_arrow_dataset"""
    truth_data = TruthData(
        self.scalar_data + self.list_data,
        self.scalar_dtypes + self.list_dtypes,
        self.scalar_shapes + self.list_shapes)

    batch = self.make_record_batch(truth_data)

    # test all columns selected
    dataset = arrow_io.ArrowDataset.from_record_batches(
        batch,
        list(range(len(truth_data.output_types))),
        truth_data.output_types,
        truth_data.output_shapes)
    self.run_test_case(dataset, truth_data)

    # test column selection
    columns = (1, 3, len(truth_data.output_types) - 1)
    dataset = arrow_io.ArrowDataset.from_record_batches(
        batch,
        columns,
        tuple([truth_data.output_types[c] for c in columns]),
        tuple([truth_data.output_shapes[c] for c in columns]))
    self.run_test_case(dataset, truth_data)

    # test construction from pd.DataFrame
    df = batch.to_pandas()
    dataset = arrow_io.ArrowDataset.from_pandas(
        df, preserve_index=False)
    self.run_test_case(dataset, truth_data)

  def test_from_pandas_preserve_index(self):
    """test_from_pandas_preserve_index"""
    data_v = [
        [1.0, 2.0, 3.0],
        [0.2, 0.4, 0.8],
    ]

    truth_data = TruthData(
        data_v,
        (dtypes.float32, dtypes.float32),
        (tf.TensorShape([]), tf.TensorShape([])))

    batch = self.make_record_batch(truth_data)
    df = batch.to_pandas()
    dataset = arrow_io.ArrowDataset.from_pandas(
        df, preserve_index=True)

    # Add index column to test data to check results
    truth_data_with_index = TruthData(
        truth_data.data + [range(len(truth_data.data[0]))],
        truth_data.output_types + (dtypes.int64,),
        truth_data.output_shapes + (tf.TensorShape([]),))
    self.run_test_case(dataset, truth_data_with_index)

    # Test preserve_index again, selecting second column only
    # NOTE: need to select TruthData because `df` gets selected also
    truth_data_selected_with_index = TruthData(
        truth_data_with_index.data[1:],
        truth_data_with_index.output_types[1:],
        truth_data_with_index.output_shapes[1:])
    dataset = arrow_io.ArrowDataset.from_pandas(
        df, columns=(1,), preserve_index=True)
    self.run_test_case(dataset, truth_data_selected_with_index)

  def test_arrow_feather_dataset(self):
    """test_arrow_feather_dataset"""
    # Feather files currently do not support columns of list types
    truth_data = TruthData(self.scalar_data, self.scalar_dtypes,
                           self.scalar_shapes)

    batch = self.make_record_batch(truth_data)
    df = batch.to_pandas()

    # Create a tempfile that is deleted after tests run
    with tempfile.NamedTemporaryFile(delete=False) as f:
      write_feather(df, f)

    # test single file
    dataset = arrow_io.ArrowFeatherDataset(
        f.name,
        list(range(len(truth_data.output_types))),
        truth_data.output_types,
        truth_data.output_shapes)
    self.run_test_case(dataset, truth_data)

    # test multiple files
    dataset = arrow_io.ArrowFeatherDataset(
        [f.name, f.name],
        list(range(len(truth_data.output_types))),
        truth_data.output_types,
        truth_data.output_shapes)
    truth_data_doubled = TruthData(
        [d * 2 for d in truth_data.data],
        truth_data.output_types,
        truth_data.output_shapes)
    self.run_test_case(dataset, truth_data_doubled)

    # test construction from schema
    dataset = arrow_io.ArrowFeatherDataset.from_schema(
        f.name, batch.schema)
    self.run_test_case(dataset, truth_data)

    os.unlink(f.name)

  def test_arrow_socket_dataset(self):
    """test_arrow_socket_dataset"""
    truth_data = TruthData(
        self.scalar_data + self.list_data,
        self.scalar_dtypes + self.list_dtypes,
        self.scalar_shapes + self.list_shapes)

    batch = self.make_record_batch(truth_data)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(1)
    host_addr, port = sock.getsockname()
    host = "%s:%s" % (host_addr, port)

    def run_server(num_batches):
      conn, _ = sock.accept()
      outfile = conn.makefile(mode='wb')
      writer = pa.RecordBatchStreamWriter(outfile, batch.schema)
      for _ in range(num_batches):
        writer.write_batch(batch)
      writer.close()
      outfile.close()
      conn.close()
      sock.close()

    # test with multiple batches, construct from schema
    num_batches = 2
    server = threading.Thread(target=run_server, args=(num_batches,))
    server.start()

    dataset = arrow_io.ArrowStreamDataset.from_schema(
        host, batch.schema)
    truth_data_mult = TruthData(
        [d * num_batches for d in truth_data.data],
        truth_data.output_types,
        truth_data.output_shapes)
    self.run_test_case(dataset, truth_data_mult)

    server.join()

  def test_arrow_unix_socket_dataset(self):
    """test_arrow_unix_socket_dataset"""
    if os.name == "nt":
      self.skipTest("Unix Domain Sockets not supported on Windows")

    truth_data = TruthData(
        self.scalar_data + self.list_data,
        self.scalar_dtypes + self.list_dtypes,
        self.scalar_shapes + self.list_shapes)

    batch = self.make_record_batch(truth_data)

    host = os.path.join(tempfile.gettempdir(), 'arrow_io_stream')

    # Make sure the socket does not already exist
    try:
      os.unlink(host)
    except OSError:
      if os.path.exists(host):
        raise

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(host)
    sock.listen(1)

    def run_server(num_batches):
      conn, _ = sock.accept()
      outfile = conn.makefile(mode='wb')
      writer = pa.RecordBatchStreamWriter(outfile, batch.schema)
      for _ in range(num_batches):
        writer.write_batch(batch)
      writer.close()
      outfile.close()
      conn.close()
      sock.close()

    # test with multiple batches, construct from schema
    num_batches = 2
    server = threading.Thread(target=run_server, args=(num_batches,))
    server.start()

    endpoint = 'unix://{}'.format(host)

    dataset = arrow_io.ArrowStreamDataset.from_schema(
        endpoint, batch.schema)
    truth_data_mult = TruthData(
        [d * num_batches for d in truth_data.data],
        truth_data.output_types,
        truth_data.output_shapes)
    self.run_test_case(dataset, truth_data_mult)

    server.join()

  def test_multiple_stream_hosts(self):
    """test_multiple_stream_hosts"""
    if os.name == "nt":
      self.skipTest("Unix Domain Sockets not supported on Windows")

    truth_data = TruthData(
        self.scalar_data + self.list_data,
        self.scalar_dtypes + self.list_dtypes,
        self.scalar_shapes + self.list_shapes)

    batch = self.make_record_batch(truth_data)

    hosts = [os.path.join(tempfile.gettempdir(), 'arrow_io_stream_{}'.format(i))
             for i in range(1, 3)]

    def start_server(host):
      """start_server"""
      try:
        os.unlink(host)
      except OSError:
        if os.path.exists(host):
          raise

      sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
      sock.bind(host)
      sock.listen(1)

      def run_server(num_batches):
        """run_server"""
        conn, _ = sock.accept()
        outfile = conn.makefile(mode='wb')
        writer = pa.RecordBatchStreamWriter(outfile, batch.schema)
        for _ in range(num_batches):
          writer.write_batch(batch)
        writer.close()
        outfile.close()
        conn.close()
        sock.close()

      # test with multiple batches, construct from schema
      server = threading.Thread(target=run_server, args=(1,))
      server.start()
      return server

    servers = [start_server(h) for h in hosts]
    endpoints = ['unix://{}'.format(h) for h in hosts]

    dataset = arrow_io.ArrowStreamDataset.from_schema(
        endpoints, batch.schema)
    truth_data_mult = TruthData(
        [d * len(hosts) for d in truth_data.data],
        truth_data.output_types,
        truth_data.output_shapes)
    self.run_test_case(dataset, truth_data_mult)

    for s in servers:
      s.join()

  def test_stream_from_pandas(self):
    """test_stream_from_pandas"""

    truth_data = TruthData(
        self.scalar_data,
        self.scalar_dtypes,
        self.scalar_shapes)

    batch = self.make_record_batch(truth_data)
    df = batch.to_pandas()

    batch_size = 2

    # Test preserve index False
    dataset = arrow_io.ArrowStreamDataset.from_pandas(
        df,
        batch_size=batch_size,
        preserve_index=False)
    self.run_test_case(dataset, truth_data, batch_size=batch_size)

    # Test preserve index True and select all but index columns
    truth_data = TruthData(
        truth_data.data + [range(len(truth_data.data[0]))],
        truth_data.output_types + (dtypes.int64,),
        truth_data.output_shapes + (tf.TensorShape([]),))
    dataset = arrow_io.ArrowStreamDataset.from_pandas(
        df,
        batch_size=batch_size,
        preserve_index=True)
    self.run_test_case(dataset, truth_data, batch_size=batch_size)

  def test_stream_from_pandas_remainder(self):
    """Test stream from Pandas that produces partial batch"""
    batch_size = len(self.scalar_data[0]) - 1

    truth_data = TruthData(
        self.scalar_data,
        self.scalar_dtypes,
        self.scalar_shapes)

    batch = self.make_record_batch(truth_data)
    df = batch.to_pandas()

    dataset = arrow_io.ArrowStreamDataset.from_pandas(
        df,
        batch_size=batch_size,
        preserve_index=False)
    self.run_test_case(dataset, truth_data, batch_size=batch_size)

  def test_stream_from_pandas_iter(self):
    """test_stream_from_pandas_iter"""

    batch_data = TruthData(
        self.scalar_data,
        self.scalar_dtypes,
        self.scalar_shapes)

    batch = self.make_record_batch(batch_data)
    df = batch.to_pandas()

    batch_size = 2
    num_iters = 3

    dataset = arrow_io.ArrowStreamDataset.from_pandas(
        (df for _ in range(num_iters)),
        batch_size=batch_size,
        preserve_index=False)

    truth_data = TruthData(
        [d * num_iters for d in batch_data.data],
        batch_data.output_types,
        batch_data.output_shapes)

    self.run_test_case(dataset, truth_data, batch_size=batch_size)

  def test_stream_from_pandas_not_batched(self):
    """test_stream_from_pandas_not_batched"""

    truth_data = TruthData(
        self.scalar_data,
        self.scalar_dtypes,
        self.scalar_shapes)

    batch = self.make_record_batch(truth_data)
    df = batch.to_pandas()

    dataset = arrow_io.ArrowStreamDataset.from_pandas(
        df,
        preserve_index=False)
    self.run_test_case(dataset, truth_data)

  def test_bool_array_type(self):
    """
    NOTE: need to test this seperately because to_pandas fails with
    ArrowNotImplementedError:
      Not implemented type for list in DataFrameBlock: bool
    see https://issues.apache.org/jira/browse/ARROW-4370
    """
    truth_data = TruthData(
        [[[False, False], [False, True], [True, False], [True, True]]],
        (dtypes.bool,),
        (tf.TensorShape([None]),))

    batch = self.make_record_batch(truth_data)

    dataset = arrow_io.ArrowDataset.from_record_batches(
        batch,
        (0,),
        truth_data.output_types,
        truth_data.output_shapes)
    self.run_test_case(dataset, truth_data)

  def test_incorrect_column_type(self):
    """Test that a column with incorrect dtype raises error"""
    truth_data = TruthData(self.scalar_data, self.scalar_dtypes,
                           self.scalar_shapes)
    batch = self.make_record_batch(truth_data)

    dataset = arrow_io.ArrowDataset.from_record_batches(
        batch,
        list(range(len(truth_data.output_types))),
        tuple([dtypes.int32 for _ in truth_data.output_types]),
        truth_data.output_shapes)
    with self.assertRaisesRegex(errors.OpError, 'Arrow type mismatch'):
      self.run_test_case(dataset, truth_data)

  def test_map_and_batch(self):
    """Test that using map then batch produces correct output. This will create
    a map_and_batch_dataset_op that calls GetNext after end_of_sequence=true
    """
    truth_data = TruthData(
        [list(range(10))],
        (dtypes.int32,),
        (tf.TensorShape([]),))
    batch = self.make_record_batch(truth_data)
    dataset = arrow_io.ArrowDataset.from_record_batches(
        batch,
        list(range(len(truth_data.output_types))),
        truth_data.output_types,
        truth_data.output_shapes)

    dataset = dataset.map(lambda x: x).batch(4)

    expected = truth_data.data[0]
    for result_tensors in dataset:
      results = result_tensors.numpy()
      for x in results:
        self.assertTrue(expected, 'Dataset has more output than expected')
        self.assertEqual(x, expected[0])
        expected.pop(0)

  def test_tf_function(self):
    """ Test that an ArrowDataset can be used in tf.function call
    """
    if not tf.version.VERSION.startswith("2."):
      self.skipTest("Test requires TF2.0 for tf.function")

    truth_data = TruthData(
        [list(range(10)), [x * 1.1 for x in range(10)]],
        (dtypes.int32, dtypes.float64),
        (tf.TensorShape([]), tf.TensorShape([])))

    @tf.function
    def create_arrow_dataset(serialized_batch):
      """Create an arrow dataset from input tensor"""
      dataset = arrow_io.ArrowDataset(
          serialized_batch,
          list(range(len(truth_data.output_types))),
          truth_data.output_types,
          truth_data.output_shapes)
      return dataset

    batch = self.make_record_batch(truth_data)
    buf = io.BytesIO()
    writer = pa.RecordBatchFileWriter(buf, batch.schema)
    writer.write_batch(batch)
    writer.close()

    for row, results in enumerate(create_arrow_dataset(buf.getvalue())):
      value = [result.numpy() for result in results]
      self.assertEqual(value[0], truth_data.data[0][row])
      self.assertAlmostEqual(value[1], truth_data.data[1][row], 4)

  def test_batch_no_remainder(self):
    """Test batch_size that does not leave a remainder
    """
    batch_size = len(self.scalar_data[0])
    num_batches = 2

    truth_data = TruthData(
        [d * num_batches for d in self.scalar_data],
        self.scalar_dtypes,
        self.scalar_shapes)

    batch = self.make_record_batch(truth_data)
    df = batch.to_pandas()

    dataset = arrow_io.ArrowDataset.from_pandas(
        df, preserve_index=False, batch_size=batch_size)
    self.run_test_case(dataset, truth_data, batch_size=batch_size)

  def test_batch_remainder(self):
    """Test batch_size that does leave a remainder
    """
    batch_size = len(self.scalar_data[0]) - 1

    truth_data = TruthData(
        self.scalar_data,
        self.scalar_dtypes,
        self.scalar_shapes)

    batch = self.make_record_batch(truth_data)
    df = batch.to_pandas()

    dataset = arrow_io.ArrowDataset.from_pandas(
        df, preserve_index=False, batch_size=batch_size)
    self.run_test_case(dataset, truth_data, batch_size=batch_size)

  def test_batch_drop_remainder(self):
    """Test batch_size that drops remainder data
    """
    batch_size = len(self.scalar_data[0]) - 1

    truth_data = TruthData(
        self.scalar_data,
        self.scalar_dtypes,
        self.scalar_shapes)

    batch = self.make_record_batch(truth_data)
    df = batch.to_pandas()

    truth_data_drop_last = TruthData(
        [d[:-1] for d in truth_data.data],
        truth_data.output_types,
        truth_data.output_shapes)

    dataset = arrow_io.ArrowDataset.from_pandas(
        df,
        preserve_index=False,
        batch_size=batch_size,
        batch_mode='drop_remainder')
    self.run_test_case(dataset, truth_data_drop_last, batch_size=batch_size)

  def test_batch_mode_auto(self):
    """Test auto batch_mode to size to record batch number of rows
    """
    num_batches = 2

    single_batch_data = TruthData(
        self.scalar_data,
        self.scalar_dtypes,
        self.scalar_shapes)

    batch = self.make_record_batch(single_batch_data)
    batches = [batch] * num_batches

    truth_data = TruthData(
        [d * num_batches for d in single_batch_data.data],
        single_batch_data.output_types,
        single_batch_data.output_shapes)

    dataset = arrow_io.ArrowDataset.from_record_batches(
        batches,
        list(range(len(truth_data.output_types))),
        truth_data.output_types,
        truth_data.output_shapes,
        batch_mode='auto')

    self.run_test_case(dataset, truth_data, batch_size=batch.num_rows)

  def test_batch_with_partials(self):
    """Test batch_size that divides an Arrow record batch into partial batches
    """
    num_batches = 3
    batch_size = int(len(self.scalar_data[0]) * 1.5)

    single_batch_data = TruthData(
        self.scalar_data,
        self.scalar_dtypes,
        self.scalar_shapes)

    batch = self.make_record_batch(single_batch_data)
    batches = [batch] * num_batches

    truth_data = TruthData(
        [d * num_batches for d in single_batch_data.data],
        single_batch_data.output_types,
        single_batch_data.output_shapes)

    # Batches should divide input without remainder
    self.assertEqual(len(truth_data.data[0]) % batch_size, 0)

    dataset = arrow_io.ArrowDataset.from_record_batches(
        batches,
        list(range(len(truth_data.output_types))),
        truth_data.output_types,
        truth_data.output_shapes,
        batch_size=batch_size)

    self.run_test_case(dataset, truth_data, batch_size=batch_size)

  def test_batch_with_partials_and_remainder(self):
    """ Test batch_size that divides an Arrow record batch into partial batches
    and leaves remainder data
    """
    num_batches = 3
    batch_size = len(self.scalar_data[0]) + 1

    single_batch_data = TruthData(
        self.scalar_data,
        self.scalar_dtypes,
        self.scalar_shapes)

    batch = self.make_record_batch(single_batch_data)
    batches = [batch] * num_batches

    truth_data = TruthData(
        [d * num_batches for d in single_batch_data.data],
        single_batch_data.output_types,
        single_batch_data.output_shapes)

    # Batches should divide input and leave a remainder
    self.assertNotEqual(len(truth_data.data[0]) % batch_size, 0)

    dataset = arrow_io.ArrowDataset.from_record_batches(
        batches,
        list(range(len(truth_data.output_types))),
        truth_data.output_types,
        truth_data.output_shapes,
        batch_size=batch_size)

    self.run_test_case(dataset, truth_data, batch_size=batch_size)

  def test_batch_spans_mulitple_partials(self):
    """Test large batch_size that spans mulitple Arrow record batches
    """
    num_batches = 6
    batch_size = int(len(self.scalar_data[0]) * 3)

    single_batch_data = TruthData(
        self.scalar_data,
        self.scalar_dtypes,
        self.scalar_shapes)

    batch = self.make_record_batch(single_batch_data)
    batches = [batch] * num_batches

    truth_data = TruthData(
        [d * num_batches for d in single_batch_data.data],
        single_batch_data.output_types,
        single_batch_data.output_shapes)

    dataset = arrow_io.ArrowDataset.from_record_batches(
        batches,
        list(range(len(truth_data.output_types))),
        truth_data.output_types,
        truth_data.output_shapes,
        batch_size=batch_size)

    self.run_test_case(dataset, truth_data, batch_size=batch_size)

  def test_batch_fixed_lists(self):
    """Test batching with fixed length list types
    """
    batch_size = int(len(self.list_data[0]) / 2)

    fixed_width_list_idx = [0, 2, 3, 5]

    truth_data = TruthData(
        [self.list_data[i] for i in fixed_width_list_idx],
        tuple([self.list_dtypes[i] for i in fixed_width_list_idx]),
        tuple([self.list_shapes[i] for i in fixed_width_list_idx]))

    batch = self.make_record_batch(truth_data)

    dataset = arrow_io.ArrowDataset.from_record_batches(
        [batch],
        list(range(len(truth_data.output_types))),
        truth_data.output_types,
        truth_data.output_shapes,
        batch_size=batch_size)

    self.run_test_case(dataset, truth_data, batch_size=batch_size)

  def test_batch_variable_length_list(self):
    """Test batching with variable length lists raises error
    """
    batch_size = len(self.list_data[1])

    truth_data = TruthData(
        [self.list_data[1]],
        (self.list_dtypes[1],),
        (self.list_shapes[1],))

    batch = self.make_record_batch(truth_data)

    dataset = arrow_io.ArrowDataset.from_record_batches(
        [batch],
        list(range(len(truth_data.output_types))),
        truth_data.output_types,
        truth_data.output_shapes,
        batch_size=batch_size)

    with self.assertRaisesRegex(errors.OpError, 'variable.*unsupported'):
      self.run_test_case(dataset, truth_data, batch_size=batch_size)

  def test_unsupported_batch_mode(self):
    """Test using an unsupported batch mode
    """
    truth_data = TruthData(
        self.scalar_data,
        self.scalar_dtypes,
        self.scalar_shapes)

    with self.assertRaisesRegex(ValueError, 'Unsupported batch_mode.*doh'):
      arrow_io.ArrowDataset.from_record_batches(
          [self.make_record_batch(truth_data)],
          list(range(len(truth_data.output_types))),
          truth_data.output_types,
          truth_data.output_shapes,
          batch_mode='doh')


if __name__ == "__main__":
  test.main()
