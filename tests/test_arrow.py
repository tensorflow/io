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


from collections import namedtuple
import io
import os
import socket
import tempfile
import threading
import pytest

import pyarrow as pa

import numpy.testing as npt

import tensorflow as tf
import tensorflow_io as tfio


TruthData = namedtuple("TruthData", ["data", "output_types", "output_shapes"])


class ArrowTestBase(tf.test.TestCase):
    """ArrowTestBase"""

    @classmethod
    def setUpClass(cls):  # pylint: disable=invalid-name
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
            tf.dtypes.bool,
            tf.dtypes.int8,
            tf.dtypes.int16,
            tf.dtypes.int32,
            tf.dtypes.int64,
            tf.dtypes.uint8,
            tf.dtypes.uint16,
            tf.dtypes.uint32,
            tf.dtypes.uint64,
            tf.dtypes.float32,
            tf.dtypes.float64,
        )
        cls.scalar_shapes = tuple([tf.TensorShape([]) for _ in cls.scalar_dtypes])

        cls.list_fixed_data = [
            [[1, 1], [2, 2], [3, 3], [4, 4]],
            [[1, 1], [2, 2], [3, 3], [4, 4]],
            [[1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4]],
            [[1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4]],
        ]
        cls.list_fixed_dtypes = (
            tf.dtypes.int32,
            tf.dtypes.int64,
            tf.dtypes.float32,
            tf.dtypes.float64,
        )
        cls.list_fixed_shapes = tuple(
            [tf.TensorShape([None]) for _ in cls.list_fixed_dtypes]
        )

        cls.list_var_data = [
            [[1], [2, 2], [3, 3, 3], [4, 4, 4]],
            [[1.1], [2.2, 2.2], [3.3, 3.3, 3.3], [4.4, 4.4, 4.4]],
        ]
        cls.list_var_dtypes = (tf.dtypes.int32, tf.dtypes.float32)
        cls.list_var_shapes = (tf.TensorShape([None]), tf.TensorShape([None]))

        cls.list_data = cls.list_fixed_data + cls.list_var_data
        cls.list_dtypes = cls.list_fixed_dtypes + cls.list_var_dtypes
        cls.list_shapes = cls.list_fixed_shapes + cls.list_var_shapes

    def get_arrow_type(self, dt, is_list):
        """get_arrow_type"""
        if dt == tf.dtypes.bool:
            arrow_type = pa.bool_()
        elif dt == tf.dtypes.int8:
            arrow_type = pa.int8()
        elif dt == tf.dtypes.int16:
            arrow_type = pa.int16()
        elif dt == tf.dtypes.int32:
            arrow_type = pa.int32()
        elif dt == tf.dtypes.int64:
            arrow_type = pa.int64()
        elif dt == tf.dtypes.uint8:
            arrow_type = pa.uint8()
        elif dt == tf.dtypes.uint16:
            arrow_type = pa.uint16()
        elif dt == tf.dtypes.uint32:
            arrow_type = pa.uint32()
        elif dt == tf.dtypes.uint64:
            arrow_type = pa.uint64()
        elif dt == tf.dtypes.float16:
            arrow_type = pa.float16()
        elif dt == tf.dtypes.float32:
            arrow_type = pa.float32()
        elif dt == tf.dtypes.float64:
            arrow_type = pa.float64()
        elif dt == tf.dtypes.string:
            arrow_type = pa.string()
        else:
            raise TypeError("Unsupported dtype for Arrow" + str(dt))
        if is_list:
            arrow_type = pa.list_(arrow_type)
        return arrow_type

    def make_record_batch(self, truth_data):
        """Make an Arrow RecordBatch for given test data"""
        arrays = [
            pa.array(
                truth_data.data[col],
                type=self.get_arrow_type(
                    truth_data.output_types[col],
                    isinstance(truth_data.data[col][0], list),
                ),
            )
            for col in range(len(truth_data.output_types))
        ]
        names = ["{}_[{}]".format(i, a.type) for i, a in enumerate(arrays)]
        return pa.RecordBatch.from_arrays(arrays, names)


class ArrowIOTensorTest(ArrowTestBase):
    """ArrowIOTensorTest"""

    @classmethod
    def setUpClass(cls):  # pylint: disable=invalid-name
        """setUpClass"""
        super().setUpClass()
        cls.scalar_shapes = tuple([tf.TensorShape([len(c)]) for c in cls.scalar_data])
        cls.list_fixed_shapes = tuple(
            [tf.TensorShape([len(c), len(c[0])]) for c in cls.list_fixed_data]
        )

    def make_table(self, truth_data):
        """make_table"""
        batch = self.make_record_batch(truth_data)
        return pa.Table.from_batches([batch])

    def run_test_case(self, iot, truth_data, columns):
        """run_test_case"""
        self.assertEqual(iot.columns, columns)
        for i, column in enumerate(columns):
            iot_col = iot(column)
            self.assertEqual(iot_col.dtype, truth_data.output_types[i])
            self.assertEqual(iot_col.shape, truth_data.output_shapes[i])
            npt.assert_almost_equal(iot_col.to_tensor().numpy(), truth_data.data[i])

    def test_arrow_io_tensor_scalar(self):
        """test_arrow_io_tensor_scalar"""
        truth_data = TruthData(self.scalar_data, self.scalar_dtypes, self.scalar_shapes)

        table = self.make_table(truth_data)
        iot = tfio.IOTensor.from_arrow(table)
        self.run_test_case(iot, truth_data, table.column_names)

    def test_arrow_io_tensor_lists(self):
        """test_arrow_io_tensor_lists"""
        truth_data = TruthData(
            self.list_fixed_data, self.list_fixed_dtypes, self.list_fixed_shapes
        )

        table = self.make_table(truth_data)
        iot = tfio.IOTensor.from_arrow(table)
        self.run_test_case(iot, truth_data, table.column_names)

    def test_arrow_io_tensor_mixed(self):
        """test_arrow_io_tensor_mixed"""
        truth_data = TruthData(
            self.scalar_data + self.list_fixed_data,
            self.scalar_dtypes + self.list_fixed_dtypes,
            self.scalar_shapes + self.list_fixed_shapes,
        )

        table = self.make_table(truth_data)
        iot = tfio.IOTensor.from_arrow(table)
        self.run_test_case(iot, truth_data, table.column_names)

    def test_arrow_io_tensor_chunked(self):
        """test_arrow_io_tensor_chunked"""

        num_chunks = 2

        chunk_data = TruthData(
            self.scalar_data + self.list_fixed_data,
            self.scalar_dtypes + self.list_fixed_dtypes,
            self.scalar_shapes + self.list_fixed_shapes,
        )

        # Make a table with double the data for 2 chunks
        table = self.make_table(chunk_data)
        table = pa.concat_tables([table] * num_chunks)

        # Double the batch size of the truth data
        output_shapes = self.scalar_shapes + self.list_fixed_shapes
        output_shapes = [
            tf.TensorShape([d + d if i == 0 else d for i, d in enumerate(shape)])
            for shape in output_shapes
        ]

        truth_data = TruthData(
            [d * num_chunks for d in chunk_data.data],
            self.scalar_dtypes + self.list_fixed_dtypes,
            output_shapes,
        )

        self.assertGreater(table[0].num_chunks, 1)
        iot = tfio.IOTensor.from_arrow(table)
        self.run_test_case(iot, truth_data, table.column_names)

    def test_arrow_io_dataset_map_from_file(self):
        """test_arrow_io_dataset_map_from_file"""
        column = "a"
        dtype = tf.dtypes.int64
        column_dtype = self.get_arrow_type(dtype, False)
        arr = pa.array(list(range(100)), column_dtype)
        table = pa.Table.from_arrays([arr], [column])
        spec = {column: dtype}

        with tempfile.NamedTemporaryFile(delete=False) as f:
            with pa.RecordBatchFileWriter(f.name, table.schema) as writer:
                for batch in table.to_batches():
                    writer.write_batch(batch)

        def from_file(_):
            reader = pa.RecordBatchFileReader(f.name)
            t = reader.read_all()
            tio = tfio.IOTensor.from_arrow(t, spec=spec)
            return tio(column).to_tensor()

        num_iters = 2
        ds = tf.data.Dataset.range(num_iters).map(from_file)
        expected = table[column].to_pylist()

        iter_count = 0
        for result in ds:
            npt.assert_array_equal(result, expected)
            iter_count += 1

        self.assertEqual(iter_count, num_iters)
        os.unlink(f.name)

    def test_arrow_io_dataset_map_py_func(self):
        """test_arrow_io_dataset_map_from_py_func"""
        column = "a"
        dtype = tf.dtypes.int64
        column_dtype = self.get_arrow_type(dtype, False)
        arr = pa.array(list(range(100)), column_dtype)
        table = pa.Table.from_arrays([arr], [column])
        spec = {column: dtype}

        with tempfile.NamedTemporaryFile(delete=False) as f:
            with pa.RecordBatchFileWriter(f.name, table.schema) as writer:
                for batch in table.to_batches():
                    writer.write_batch(batch)

        def read_table(filename):
            filename = filename.numpy().decode("utf-8")
            reader = pa.RecordBatchFileReader(filename)
            return reader.read_all()

        def from_py_func(filename):
            from tensorflow_io.python.ops.arrow_io_tensor_ops import ArrowIOResource

            table_res = ArrowIOResource.from_py_function(read_table, [filename])
            tio = tfio.IOTensor.from_arrow(table_res, spec=spec)
            return tio(column).to_tensor()

        num_iters = 2
        ds = tf.data.Dataset.from_tensor_slices([f.name, f.name]).map(from_py_func)
        expected = table[column].to_pylist()

        iter_count = 0
        for result in ds:
            npt.assert_array_equal(result, expected)
            iter_count += 1

        self.assertEqual(iter_count, num_iters)
        os.unlink(f.name)

    def test_spec_selection_by_column_name(self):
        """test_spec_selection_by_column_name"""

        def from_func(_):
            a = pa.array([1, 2, 3], type=pa.int32())
            b = pa.array([4, 5, 6], type=pa.int64())
            c = pa.array([7, 8, 9], type=pa.float32())
            t = pa.Table.from_arrays([a, b, c], ["a", "b", "c"])
            foo = tfio.IOTensor.from_arrow(t, spec={"b": tf.int64})
            return foo("b").to_tensor()

        ds = tf.data.Dataset.range(1).map(from_func)
        results = list(ds.as_numpy_iterator())
        self.assertEqual(len(results), 1)
        result = results[0]

        b = pa.array([4, 5, 6], type=pa.int64())
        expected = b.to_numpy()

        npt.assert_array_equal(result, expected)

    def test_spec_selection_by_column_index(self):
        """test_spec_selection_by_column_index"""

        def from_func(_):
            a = pa.array([1, 2, 3], type=pa.int32())
            b = pa.array([4, 5, 6], type=pa.int64())
            c = pa.array([7, 8, 9], type=pa.float32())
            t = pa.Table.from_arrays([a, b, c], ["a", "b", "c"])
            foo = tfio.IOTensor.from_arrow(t, spec={1: tf.int64})
            return foo(1).to_tensor()

        ds = tf.data.Dataset.range(1).map(from_func)
        results = list(ds.as_numpy_iterator())
        self.assertEqual(len(results), 1)
        result = results[0]

        b = pa.array([4, 5, 6], type=pa.int64())
        expected = b.to_numpy()

        npt.assert_array_equal(result, expected)


class ArrowDatasetTest(ArrowTestBase):
    """ArrowDatasetTest"""

    def run_test_case(self, dataset, truth_data, batch_size=None):
        """run_test_case"""

        def is_float(dtype):
            """Check if dtype is a floating-point"""
            return dtype in [tf.dtypes.float16, tf.dtypes.float32, tf.dtypes.float64]

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
                        self.assertListEqual(
                            value[i].tolist(), truth_data.data[col][row]
                        )

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

    def test_arrow_dataset(self):
        """test_arrow_dataset"""
        import tensorflow_io.arrow as arrow_io

        truth_data = TruthData(
            self.scalar_data + self.list_data,
            self.scalar_dtypes + self.list_dtypes,
            self.scalar_shapes + self.list_shapes,
        )

        batch = self.make_record_batch(truth_data)

        # test all columns selected
        dataset = arrow_io.ArrowDataset.from_record_batches(
            batch, truth_data.output_types, truth_data.output_shapes
        )
        self.run_test_case(dataset, truth_data)

        # test column selection
        columns = (1, 3, len(truth_data.output_types) - 1)
        dataset = arrow_io.ArrowDataset.from_record_batches(
            batch,
            tuple([truth_data.output_types[c] for c in columns]),
            tuple([truth_data.output_shapes[c] for c in columns]),
            columns=columns,
        )
        self.run_test_case(dataset, truth_data)

        # test construction from pd.DataFrame
        df = batch.to_pandas()
        dataset = arrow_io.ArrowDataset.from_pandas(df, preserve_index=False)
        self.run_test_case(dataset, truth_data)

    def test_arrow_dataset_with_strings(self):
        """test_arrow_dataset"""
        import tensorflow_io.arrow as arrow_io

        scalar_data = [
            [b"1.1", b"2.2", b"3.3", b"4.4"],
        ]
        scalar_dtypes = (tf.dtypes.string,)
        scalar_shapes = tuple([tf.TensorShape([]) for _ in scalar_dtypes])
        truth_data = TruthData(scalar_data, scalar_dtypes, scalar_shapes)

        batch = self.make_record_batch(truth_data)

        # test all columns selected
        dataset = arrow_io.ArrowDataset.from_record_batches(
            batch, truth_data.output_types, truth_data.output_shapes
        )
        self.run_test_case(dataset, truth_data)

    def test_from_pandas_preserve_index(self):
        """test_from_pandas_preserve_index"""
        import tensorflow_io.arrow as arrow_io

        data_v = [
            [1.0, 2.0, 3.0],
            [0.2, 0.4, 0.8],
        ]

        truth_data = TruthData(
            data_v,
            (tf.dtypes.float32, tf.dtypes.float32),
            (tf.TensorShape([]), tf.TensorShape([])),
        )

        batch = self.make_record_batch(truth_data)
        df = batch.to_pandas()
        dataset = arrow_io.ArrowDataset.from_pandas(df, preserve_index=True)

        # Add index column to test data to check results
        truth_data_with_index = TruthData(
            truth_data.data + [range(len(truth_data.data[0]))],
            truth_data.output_types + (tf.dtypes.int64,),
            truth_data.output_shapes + (tf.TensorShape([]),),
        )
        self.run_test_case(dataset, truth_data_with_index)

        # Test preserve_index again, selecting second column only
        # NOTE: need to select TruthData because `df` gets selected also
        truth_data_selected_with_index = TruthData(
            truth_data_with_index.data[1:],
            truth_data_with_index.output_types[1:],
            truth_data_with_index.output_shapes[1:],
        )
        dataset = arrow_io.ArrowDataset.from_pandas(
            df, columns=(1,), preserve_index=True
        )
        self.run_test_case(dataset, truth_data_selected_with_index)

    def test_arrow_feather_dataset(self):
        """test_arrow_feather_dataset"""
        import tensorflow_io.arrow as arrow_io

        from pyarrow.feather import write_feather

        # Feather files currently do not support columns of list types
        truth_data = TruthData(self.scalar_data, self.scalar_dtypes, self.scalar_shapes)

        batch = self.make_record_batch(truth_data)
        df = batch.to_pandas()

        # Create a tempfile that is deleted after tests run
        with tempfile.NamedTemporaryFile(delete=False) as f:
            write_feather(df, f, version=1)

        # test single file
        dataset = arrow_io.ArrowFeatherDataset(
            f.name,
            list(range(len(truth_data.output_types))),
            truth_data.output_types,
            truth_data.output_shapes,
        )
        self.run_test_case(dataset, truth_data)

        # test single file with 'file://' prefix
        dataset = arrow_io.ArrowFeatherDataset(
            "file://{}".format(f.name),
            list(range(len(truth_data.output_types))),
            truth_data.output_types,
            truth_data.output_shapes,
        )
        self.run_test_case(dataset, truth_data)

        # test multiple files
        dataset = arrow_io.ArrowFeatherDataset(
            [f.name, f.name],
            list(range(len(truth_data.output_types))),
            truth_data.output_types,
            truth_data.output_shapes,
        )
        truth_data_doubled = TruthData(
            [d * 2 for d in truth_data.data],
            truth_data.output_types,
            truth_data.output_shapes,
        )
        self.run_test_case(dataset, truth_data_doubled)

        # test construction from schema
        dataset = arrow_io.ArrowFeatherDataset.from_schema(f.name, batch.schema)
        self.run_test_case(dataset, truth_data)

        os.unlink(f.name)

    def test_arrow_socket_dataset(self):
        """test_arrow_socket_dataset"""
        import tensorflow_io.arrow as arrow_io

        truth_data = TruthData(
            self.scalar_data + self.list_data,
            self.scalar_dtypes + self.list_dtypes,
            self.scalar_shapes + self.list_shapes,
        )

        batch = self.make_record_batch(truth_data)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        host_addr, port = sock.getsockname()
        host = "{}:{}".format(host_addr, port)

        def run_server(num_batches):
            conn, _ = sock.accept()
            outfile = conn.makefile(mode="wb")
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

        dataset = arrow_io.ArrowStreamDataset.from_schema(host, batch.schema)
        truth_data_mult = TruthData(
            [d * num_batches for d in truth_data.data],
            truth_data.output_types,
            truth_data.output_shapes,
        )
        self.run_test_case(dataset, truth_data_mult)

        server.join()

    def test_arrow_unix_socket_dataset(self):
        """test_arrow_unix_socket_dataset"""
        import tensorflow_io.arrow as arrow_io

        if os.name == "nt":
            self.skipTest("Unix Domain Sockets not supported on Windows")

        truth_data = TruthData(
            self.scalar_data + self.list_data,
            self.scalar_dtypes + self.list_dtypes,
            self.scalar_shapes + self.list_shapes,
        )

        batch = self.make_record_batch(truth_data)

        host = os.path.join(tempfile.gettempdir(), "arrow_io_stream")

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
            outfile = conn.makefile(mode="wb")
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

        endpoint = "unix://{}".format(host)

        dataset = arrow_io.ArrowStreamDataset.from_schema(endpoint, batch.schema)
        truth_data_mult = TruthData(
            [d * num_batches for d in truth_data.data],
            truth_data.output_types,
            truth_data.output_shapes,
        )
        self.run_test_case(dataset, truth_data_mult)

        server.join()

    def test_multiple_stream_hosts(self):
        """test_multiple_stream_hosts"""
        import tensorflow_io.arrow as arrow_io

        if os.name == "nt":
            self.skipTest("Unix Domain Sockets not supported on Windows")

        truth_data = TruthData(
            self.scalar_data + self.list_data,
            self.scalar_dtypes + self.list_dtypes,
            self.scalar_shapes + self.list_shapes,
        )

        batch = self.make_record_batch(truth_data)

        hosts = [
            os.path.join(tempfile.gettempdir(), "arrow_io_stream_{}".format(i))
            for i in range(1, 3)
        ]

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
                outfile = conn.makefile(mode="wb")
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
        endpoints = ["unix://{}".format(h) for h in hosts]

        dataset = arrow_io.ArrowStreamDataset.from_schema(endpoints, batch.schema)
        truth_data_mult = TruthData(
            [d * len(hosts) for d in truth_data.data],
            truth_data.output_types,
            truth_data.output_shapes,
        )
        self.run_test_case(dataset, truth_data_mult)

        for s in servers:
            s.join()

    def test_stream_from_pandas(self):
        """test_stream_from_pandas"""
        import tensorflow_io.arrow as arrow_io

        truth_data = TruthData(self.scalar_data, self.scalar_dtypes, self.scalar_shapes)

        batch = self.make_record_batch(truth_data)
        df = batch.to_pandas()

        batch_size = 2

        # Test preserve index False
        dataset = arrow_io.ArrowStreamDataset.from_pandas(
            df, batch_size=batch_size, preserve_index=False
        )
        self.run_test_case(dataset, truth_data, batch_size=batch_size)

        # Test preserve index True and select all but index columns
        truth_data = TruthData(
            truth_data.data + [range(len(truth_data.data[0]))],
            truth_data.output_types + (tf.dtypes.int64,),
            truth_data.output_shapes + (tf.TensorShape([]),),
        )
        dataset = arrow_io.ArrowStreamDataset.from_pandas(
            df, batch_size=batch_size, preserve_index=True
        )
        self.run_test_case(dataset, truth_data, batch_size=batch_size)

    def test_stream_from_pandas_remainder(self):
        """Test stream from Pandas that produces partial batch"""
        import tensorflow_io.arrow as arrow_io

        batch_size = len(self.scalar_data[0]) - 1

        truth_data = TruthData(self.scalar_data, self.scalar_dtypes, self.scalar_shapes)

        batch = self.make_record_batch(truth_data)
        df = batch.to_pandas()

        dataset = arrow_io.ArrowStreamDataset.from_pandas(
            df, batch_size=batch_size, preserve_index=False
        )
        self.run_test_case(dataset, truth_data, batch_size=batch_size)

    def test_stream_from_pandas_iter(self):
        """test_stream_from_pandas_iter"""
        import tensorflow_io.arrow as arrow_io

        batch_data = TruthData(self.scalar_data, self.scalar_dtypes, self.scalar_shapes)

        batch = self.make_record_batch(batch_data)
        df = batch.to_pandas()

        batch_size = 2
        num_iters = 3

        dataset = arrow_io.ArrowStreamDataset.from_pandas(
            (df for _ in range(num_iters)), batch_size=batch_size, preserve_index=False
        )

        truth_data = TruthData(
            [d * num_iters for d in batch_data.data],
            batch_data.output_types,
            batch_data.output_shapes,
        )

        self.run_test_case(dataset, truth_data, batch_size=batch_size)

    def test_stream_from_pandas_not_batched(self):
        """test_stream_from_pandas_not_batched"""
        import tensorflow_io.arrow as arrow_io

        truth_data = TruthData(self.scalar_data, self.scalar_dtypes, self.scalar_shapes)

        batch = self.make_record_batch(truth_data)
        df = batch.to_pandas()

        dataset = arrow_io.ArrowStreamDataset.from_pandas(df, preserve_index=False)
        self.run_test_case(dataset, truth_data)

    def test_stream_from_pandas_repeat(self):
        """test_stream_from_pandas_repeat"""
        import tensorflow_io.arrow as arrow_io

        batch_data = TruthData(self.scalar_data, self.scalar_dtypes, self.scalar_shapes)

        batch = self.make_record_batch(batch_data)
        df = batch.to_pandas()

        num_repeat = 10

        dataset = arrow_io.ArrowStreamDataset.from_pandas(
            df, batch_size=2, preserve_index=False
        ).repeat(num_repeat)

        # patch columns attr so run_test_case can use
        dataset.columns = list(range(len(batch_data.output_types)))

        truth_data = TruthData(
            [d * num_repeat for d in batch_data.data],
            batch_data.output_types,
            batch_data.output_shapes,
        )

        self.run_test_case(dataset, truth_data, batch_size=2)

    def test_bool_array_type(self):
        """NOTE: need to test this separately because to_pandas fails with
        ArrowNotImplementedError: Not implemented type for list in
        DataFrameBlock: bool
        see https://issues.apache.org/jira/browse/ARROW-4370
        """
        import tensorflow_io.arrow as arrow_io

        truth_data = TruthData(
            [[[False, False], [False, True], [True, False], [True, True]]],
            (tf.dtypes.bool,),
            (tf.TensorShape([None]),),
        )

        batch = self.make_record_batch(truth_data)

        dataset = arrow_io.ArrowDataset.from_record_batches(
            batch, truth_data.output_types, truth_data.output_shapes, columns=(0,)
        )
        self.run_test_case(dataset, truth_data)

    def test_incorrect_column_type(self):
        """Test that a column with incorrect dtype raises error"""
        import tensorflow_io.arrow as arrow_io

        truth_data = TruthData(self.scalar_data, self.scalar_dtypes, self.scalar_shapes)
        batch = self.make_record_batch(truth_data)

        dataset = arrow_io.ArrowDataset.from_record_batches(
            batch,
            tuple([tf.dtypes.int32 for _ in truth_data.output_types]),
            truth_data.output_shapes,
        )
        with self.assertRaisesRegex(tf.errors.OpError, "Arrow type mismatch"):
            self.run_test_case(dataset, truth_data)

    def test_map_and_batch(self):
        """Test that using map then batch produces correct output. This will create
        a map_and_batch_dataset_op that calls GetNext after end_of_sequence=true
        """
        import tensorflow_io.arrow as arrow_io

        truth_data = TruthData(
            [list(range(10))], (tf.dtypes.int32,), (tf.TensorShape([]),)
        )
        batch = self.make_record_batch(truth_data)
        dataset = arrow_io.ArrowDataset.from_record_batches(
            batch, truth_data.output_types, truth_data.output_shapes
        )

        dataset = dataset.map(lambda x: x).batch(4)

        expected = truth_data.data[0]
        for result_tensors in dataset:
            results = result_tensors.numpy()
            for x in results:
                self.assertTrue(expected, "Dataset has more output than expected")
                self.assertEqual(x, expected[0])
                expected.pop(0)

    @pytest.mark.skip(reason="TODO")
    def test_tf_function(self):
        """Test that an ArrowDataset can be used in tf.function call"""
        import tensorflow_io.arrow as arrow_io

        if not tf.version.VERSION.startswith("2."):
            self.skipTest("Test requires TF2.0 for tf.function")

        truth_data = TruthData(
            [list(range(10)), [x * 1.1 for x in range(10)]],
            (tf.dtypes.int32, tf.dtypes.float64),
            (tf.TensorShape([]), tf.TensorShape([])),
        )

        @tf.function
        def create_arrow_dataset(serialized_batch):
            """Create an arrow dataset from input tensor"""
            dataset = arrow_io.ArrowDataset(
                serialized_batch,
                list(range(len(truth_data.output_types))),
                truth_data.output_types,
                truth_data.output_shapes,
            )
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
        """Test batch_size that does not leave a remainder"""
        import tensorflow_io.arrow as arrow_io

        batch_size = len(self.scalar_data[0])
        num_batches = 2

        truth_data = TruthData(
            [d * num_batches for d in self.scalar_data],
            self.scalar_dtypes,
            self.scalar_shapes,
        )

        batch = self.make_record_batch(truth_data)
        df = batch.to_pandas()

        dataset = arrow_io.ArrowDataset.from_pandas(
            df, preserve_index=False, batch_size=batch_size
        )
        self.run_test_case(dataset, truth_data, batch_size=batch_size)

    def test_batch_remainder(self):
        """Test batch_size that does leave a remainder"""
        import tensorflow_io.arrow as arrow_io

        batch_size = len(self.scalar_data[0]) - 1

        truth_data = TruthData(self.scalar_data, self.scalar_dtypes, self.scalar_shapes)

        batch = self.make_record_batch(truth_data)
        df = batch.to_pandas()

        dataset = arrow_io.ArrowDataset.from_pandas(
            df, preserve_index=False, batch_size=batch_size
        )
        self.run_test_case(dataset, truth_data, batch_size=batch_size)

    def test_batch_drop_remainder(self):
        """Test batch_size that drops remainder data"""
        import tensorflow_io.arrow as arrow_io

        batch_size = len(self.scalar_data[0]) - 1

        truth_data = TruthData(self.scalar_data, self.scalar_dtypes, self.scalar_shapes)

        batch = self.make_record_batch(truth_data)
        df = batch.to_pandas()

        truth_data_drop_last = TruthData(
            [d[:-1] for d in truth_data.data],
            truth_data.output_types,
            truth_data.output_shapes,
        )

        dataset = arrow_io.ArrowDataset.from_pandas(
            df, preserve_index=False, batch_size=batch_size, batch_mode="drop_remainder"
        )
        self.run_test_case(dataset, truth_data_drop_last, batch_size=batch_size)

    def test_batch_mode_auto(self):
        """Test auto batch_mode to size to record batch number of rows"""
        import tensorflow_io.arrow as arrow_io

        num_batches = 2

        single_batch_data = TruthData(
            self.scalar_data, self.scalar_dtypes, self.scalar_shapes
        )

        batch = self.make_record_batch(single_batch_data)
        batches = [batch] * num_batches

        truth_data = TruthData(
            [d * num_batches for d in single_batch_data.data],
            single_batch_data.output_types,
            single_batch_data.output_shapes,
        )

        dataset = arrow_io.ArrowDataset.from_record_batches(
            batches,
            truth_data.output_types,
            truth_data.output_shapes,
            batch_mode="auto",
        )

        self.run_test_case(dataset, truth_data, batch_size=batch.num_rows)

    def test_batch_with_partials(self):
        """Test batch_size that divides an Arrow record batch into
        partial batches
        """
        import tensorflow_io.arrow as arrow_io

        num_batches = 3
        batch_size = int(len(self.scalar_data[0]) * 1.5)

        single_batch_data = TruthData(
            self.scalar_data, self.scalar_dtypes, self.scalar_shapes
        )

        batch = self.make_record_batch(single_batch_data)
        batches = [batch] * num_batches

        truth_data = TruthData(
            [d * num_batches for d in single_batch_data.data],
            single_batch_data.output_types,
            single_batch_data.output_shapes,
        )

        # Batches should divide input without remainder
        self.assertEqual(len(truth_data.data[0]) % batch_size, 0)

        dataset = arrow_io.ArrowDataset.from_record_batches(
            batches,
            truth_data.output_types,
            truth_data.output_shapes,
            batch_size=batch_size,
        )

        self.run_test_case(dataset, truth_data, batch_size=batch_size)

    def test_batch_with_partials_and_remainder(self):
        """Test batch_size that divides an Arrow record batch into
        partial batches and leaves remainder data
        """
        import tensorflow_io.arrow as arrow_io

        num_batches = 3
        batch_size = len(self.scalar_data[0]) + 1

        single_batch_data = TruthData(
            self.scalar_data, self.scalar_dtypes, self.scalar_shapes
        )

        batch = self.make_record_batch(single_batch_data)
        batches = [batch] * num_batches

        truth_data = TruthData(
            [d * num_batches for d in single_batch_data.data],
            single_batch_data.output_types,
            single_batch_data.output_shapes,
        )

        # Batches should divide input and leave a remainder
        self.assertNotEqual(len(truth_data.data[0]) % batch_size, 0)

        dataset = arrow_io.ArrowDataset.from_record_batches(
            batches,
            truth_data.output_types,
            truth_data.output_shapes,
            batch_size=batch_size,
        )

        self.run_test_case(dataset, truth_data, batch_size=batch_size)

    def test_batch_spans_mulitple_partials(self):
        """Test large batch_size that spans mulitple Arrow record batches"""
        import tensorflow_io.arrow as arrow_io

        num_batches = 6
        batch_size = int(len(self.scalar_data[0]) * 3)

        single_batch_data = TruthData(
            self.scalar_data, self.scalar_dtypes, self.scalar_shapes
        )

        batch = self.make_record_batch(single_batch_data)
        batches = [batch] * num_batches

        truth_data = TruthData(
            [d * num_batches for d in single_batch_data.data],
            single_batch_data.output_types,
            single_batch_data.output_shapes,
        )

        dataset = arrow_io.ArrowDataset.from_record_batches(
            batches,
            truth_data.output_types,
            truth_data.output_shapes,
            batch_size=batch_size,
        )

        self.run_test_case(dataset, truth_data, batch_size=batch_size)

    def test_batch_fixed_lists(self):
        """Test batching with fixed length list types"""
        import tensorflow_io.arrow as arrow_io

        batch_size = int(len(self.list_fixed_data[0]) / 2)

        truth_data = TruthData(
            self.list_fixed_data, self.list_fixed_dtypes, self.list_fixed_shapes
        )

        batch = self.make_record_batch(truth_data)

        dataset = arrow_io.ArrowDataset.from_record_batches(
            [batch],
            truth_data.output_types,
            truth_data.output_shapes,
            batch_size=batch_size,
        )

        self.run_test_case(dataset, truth_data, batch_size=batch_size)

    def test_batch_variable_length_list_batched(self):
        """Test batching with variable length lists raises error"""
        import tensorflow_io.arrow as arrow_io

        batch_size = len(self.list_var_data[1])

        truth_data = TruthData(
            self.list_var_data, self.list_var_dtypes, self.list_var_shapes
        )

        batch = self.make_record_batch(truth_data)

        dataset = arrow_io.ArrowDataset.from_record_batches(
            [batch],
            truth_data.output_types,
            truth_data.output_shapes,
            batch_size=batch_size,
        )

        with self.assertRaisesRegex(tf.errors.OpError, "variable.*unsupported"):
            self.run_test_case(dataset, truth_data, batch_size=batch_size)

    def test_batch_variable_length_list_unbatched(self):
        """Test unbatched variable length lists"""
        import tensorflow_io.arrow as arrow_io

        batch_size = None

        truth_data = TruthData(
            self.list_var_data, self.list_var_dtypes, self.list_var_shapes
        )

        batch = self.make_record_batch(truth_data)

        dataset = arrow_io.ArrowDataset.from_record_batches(
            [batch],
            truth_data.output_types,
            truth_data.output_shapes,
            batch_size=batch_size,
        )

        self.run_test_case(dataset, truth_data, batch_size=batch_size)

    def test_unsupported_batch_mode(self):
        """Test using an unsupported batch mode"""
        import tensorflow_io.arrow as arrow_io

        truth_data = TruthData(self.scalar_data, self.scalar_dtypes, self.scalar_shapes)

        with self.assertRaisesRegex(ValueError, "Unsupported batch_mode.*doh"):
            arrow_io.ArrowDataset.from_record_batches(
                [self.make_record_batch(truth_data)],
                truth_data.output_types,
                truth_data.output_shapes,
                batch_mode="doh",
            )

    def test_arrow_list_feather_columns(self):
        """test_arrow_list_feather_columns"""
        import tensorflow_io.arrow as arrow_io

        from pyarrow.feather import write_feather

        # Feather files currently do not support columns of list types
        truth_data = TruthData(self.scalar_data, self.scalar_dtypes, self.scalar_shapes)

        batch = self.make_record_batch(truth_data)
        df = batch.to_pandas()

        # Create a tempfile that is deleted after tests run
        with tempfile.NamedTemporaryFile(delete=False) as f:
            write_feather(df, f, version=1)

        # test single file
        # prefix "file://" to test scheme file system (e.g., s3, gcs, azfs, ignite)
        columns = arrow_io.list_feather_columns("file://" + f.name)
        for name, dtype in list(zip(batch.schema.names, batch.schema.types)):
            assert columns[name].name == name
            assert columns[name].dtype == dtype
            assert columns[name].shape == [4]

        # test memory
        with open(f.name, "rb") as ff:
            memory = ff.read()
        # when memory is provided filename doesn't matter:
        columns = arrow_io.list_feather_columns("file:///non_exist", memory=memory)
        for name, dtype in list(zip(batch.schema.names, batch.schema.types)):
            assert columns[name].name == name
            assert columns[name].dtype == dtype
            assert columns[name].shape == [4]

        os.unlink(f.name)


if __name__ == "__main__":
    test.main()
