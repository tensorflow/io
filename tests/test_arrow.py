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

import tensorflow
tensorflow.compat.v1.disable_eager_execution()

from tensorflow import dtypes # pylint: disable=wrong-import-position
from tensorflow import errors # pylint: disable=wrong-import-position
from tensorflow import test   # pylint: disable=wrong-import-position
from tensorflow.compat.v1 import data # pylint: disable=wrong-import-position

import tensorflow_io.arrow as arrow_io # pylint: disable=wrong-import-position

if sys.version_info == (3, 4):
  pytest.skip(
      "pyarrow is not supported with python 3.4", allow_module_level=True)

import pyarrow as pa  # pylint: disable=wrong-import-position
from pyarrow.feather import write_feather # pylint: disable=wrong-import-position


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
        [tensorflow.TensorShape([]) for _ in cls.scalar_dtypes])

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
        [tensorflow.TensorShape([None]) for _ in cls.list_dtypes])

  def run_test_case(self, dataset, truth_data):
    """run_test_case"""
    iterator = data.make_one_shot_iterator(dataset)
    next_element = iterator.get_next()

    def is_float(dtype):
      return dtype in [dtypes.float16, dtypes.float32, dtypes.float64]

    with self.test_session() as sess:
      for row in range(len(truth_data.data[0])):
        value = sess.run(next_element)
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
        (tensorflow.TensorShape([]), tensorflow.TensorShape([])))

    batch = self.make_record_batch(truth_data)
    df = batch.to_pandas()
    dataset = arrow_io.ArrowDataset.from_pandas(
        df, preserve_index=True)

    # Add index column to test data to check results
    truth_data_with_index = TruthData(
        truth_data.data + [range(len(truth_data.data[0]))],
        truth_data.output_types + (dtypes.int64,),
        truth_data.output_shapes + (tensorflow.TensorShape([]),))
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
        (tensorflow.TensorShape([None]),))

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
    with self.assertRaisesRegexp(errors.OpError, 'Arrow type mismatch'):
      self.run_test_case(dataset, truth_data)

  def test_map_and_batch(self):
    """
    Test that using map then batch produces correct output. This will create
    a map_and_batch_dataset_op that calls GetNext after end_of_sequence=true
    """
    truth_data = TruthData(
        [list(range(10))],
        (dtypes.int32,),
        (tensorflow.TensorShape([]),))
    batch = self.make_record_batch(truth_data)
    dataset = arrow_io.ArrowDataset.from_record_batches(
        batch,
        list(range(len(truth_data.output_types))),
        truth_data.output_types,
        truth_data.output_shapes)

    dataset = dataset.map(lambda x: x).batch(4)
    it = dataset.make_one_shot_iterator()
    d = it.get_next()

    expected = truth_data.data[0]
    with self.test_session() as sess:
      while True:
        try:
          result = sess.run(d)
          self.assertTrue(expected, 'Dataset has more output than expected')
          for x in result:
            self.assertEqual(x, expected[0])
            expected.pop(0)
        except tensorflow.errors.OutOfRangeError:
          break

  def test_feed_batches(self):
    """
    Test that an ArrowDataset can initialize an iterator to feed a placeholder
    """
    truth_data = TruthData(
        [list(range(10)), [x * 1.1 for x in range(10)]],
        (dtypes.int32, dtypes.float64),
        (tensorflow.TensorShape([]), tensorflow.TensorShape([])))
    batch = self.make_record_batch(truth_data)

    buf = io.BytesIO()
    writer = pa.RecordBatchFileWriter(buf, batch.schema)
    writer.write_batch(batch)
    writer.close()

    buf_placeholder = tensorflow.compat.v1.placeholder(
        tensorflow.dtypes.string, tensorflow.TensorShape([]))

    dataset = arrow_io.ArrowDataset(
        buf_placeholder,
        list(range(len(truth_data.output_types))),
        truth_data.output_types,
        truth_data.output_shapes)
    it = dataset.make_initializable_iterator()
    next_element = it.get_next()

    with self.test_session() as sess:
      sess.run(it.initializer, feed_dict={buf_placeholder: buf.getvalue()})
      for row in range(len(truth_data.data)):
        value = sess.run(next_element)
        self.assertEqual(value[0], truth_data.data[0][row])
        self.assertAlmostEqual(value[1], truth_data.data[1][row], 4)


if __name__ == "__main__":
  test.main()
