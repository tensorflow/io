# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for BigQuery Ops."""


import concurrent.futures
from io import BytesIO
import json
import fastavro
import numpy as np
import grpc  # pylint: disable=wrong-import-order
import tensorflow as tf  # pylint: disable=wrong-import-order

from tensorflow.python.framework import dtypes  # pylint: disable=wrong-import-order
from tensorflow.python.framework import errors  # pylint: disable=wrong-import-order
from tensorflow.python.framework import ops  # pylint: disable=wrong-import-order
from tensorflow import test  # pylint: disable=wrong-import-order
from tensorflow_io.bigquery import (
    BigQueryTestClient,
    BigQueryClient,
)  # pylint: disable=wrong-import-order

import google.cloud.bigquery_storage_v1beta1.proto.storage_pb2_grpc as storage_pb2_grpc  # pylint: disable=wrong-import-order
import google.cloud.bigquery_storage_v1beta1.proto.storage_pb2 as storage_pb2  # pylint: disable=wrong-import-order

if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
    tf.compat.v1.enable_eager_execution()


class FakeBigQueryServer(storage_pb2_grpc.BigQueryStorageServicer):
    """Fake server for Cloud BigQuery Storage API."""

    def __init__(
        self,
        avro_schema,
        rows_per_stream,
    ):
        self._project_id = ""
        self._table_id = ""
        self._dataset_id = ""
        self._streams = list()
        self._avro_schema = avro_schema
        self._rows_per_stream = rows_per_stream

        self._grpc_server = grpc.server(
            concurrent.futures.ThreadPoolExecutor(max_workers=4)
        )
        storage_pb2_grpc.add_BigQueryStorageServicer_to_server(self, self._grpc_server)
        port = self._grpc_server.add_insecure_port("localhost:0")
        self._endpoint = "localhost:" + str(port)
        print("started a fake server on :" + self._endpoint)

    def start(self):
        self._grpc_server.start()

    def stop(self):
        self._grpc_server.stop(0)

    def endpoint(self):
        return self._endpoint

    def _build_stream_name(self, stream_index):
        return (
            self._project_id
            + "/"
            + self._dataset_id
            + "/"
            + self._table_id
            + "/"
            + str(stream_index)
        )

    @staticmethod
    def serialize_to_avro(rows, schema):
        """Serializes specified rows into avro format."""
        string_output = BytesIO()
        json_schema = json.loads(schema)
        for row in rows:
            fastavro.schemaless_writer(string_output, json_schema, row)
        avro_output = string_output.getvalue()
        string_output.close()
        return avro_output

    def CreateReadSession(self, request, context):  # pylint: disable=unused-argument
        """CreateReadSession"""
        print("called CreateReadSession on a fake server")
        self._project_id = request.table_reference.project_id
        self._table_id = request.table_reference.table_id
        self._dataset_id = request.table_reference.dataset_id
        self._streams = []
        response = storage_pb2.ReadSession()
        response.avro_schema.schema = self._avro_schema
        for i in range(request.requested_streams):
            stream_name = self._build_stream_name(i)
            self._streams.append(stream_name)
            stream = response.streams.add()
            stream.name = stream_name
        return response

    def ReadRows(self, request, context):  # pylint: disable=unused-argument
        """ReadRows"""
        print("called ReadRows on a fake server: %s" % str(request))
        response = storage_pb2.ReadRowsResponse()
        stream_index = self._streams.index(request.read_position.stream.name)
        if 0 <= stream_index < len(self._rows_per_stream):
            rows = self._rows_per_stream[stream_index][request.read_position.offset :]
            serialized_rows = FakeBigQueryServer.serialize_to_avro(
                rows, self._avro_schema
            )
            response.avro_rows.serialized_binary_rows = serialized_rows
            response.avro_rows.row_count = len(rows)
        yield response


class BigqueryOpsTest(test.TestCase):
    """Tests for BigQuery adapter."""

    GCP_PROJECT_ID = "test_project_id"
    DATASET_ID = "test_dataset"
    TABLE_ID = "test_table"
    PARENT = "projects/test_parent"

    AVRO_SCHEMA = """
      {
      "type": "record",
      "name": "__root__",
      "fields": [
          {
              "name": "string",
              "type": [
                  "null",
                  "string"
              ],
              "doc": "nullable string"
          },
          {
              "name": "boolean",
              "type": [
                  "null",
                  "boolean"
              ],
              "doc": "nullable boolean"
          },
          {
              "name": "int",
              "type": [
                  "null",
                  "int"
              ],
              "doc": "nullable int"
          },
          {
              "name": "long",
              "type": [
                  "null",
                  "long"
              ],
              "doc": "nullable long"
          },
          {
              "name": "float",
              "type": [
                  "null",
                  "float"
              ],
              "doc": "nullable float"
          },
          {
              "name": "double",
              "type": [
                  "null",
                  "double"
              ],
              "doc": "nullable double"
          },
          {
              "name": "repeated_bool",
              "type": {"type": "array", "items": "boolean"},
              "doc": "repeated string"
          },
          {
              "name": "repeated_int",
              "type": {"type": "array", "items": "int"},
              "doc": "repeated string"
          },
          {
              "name": "repeated_long",
              "type": {"type": "array", "items": "long"},
              "doc": "repeated string"
          },
          {
              "name": "repeated_float",
              "type": {"type": "array", "items": "float"},
              "doc": "repeated string"
          },
          {
              "name": "repeated_double",
              "type": {"type": "array", "items": "double"},
              "doc": "repeated double"
          },
          {
              "name": "repeated_string",
              "type": {"type": "array", "items": "string"},
              "doc": "repeated string"
          }

      ]
  }"""

    STREAM_1_ROWS = [
        {
            "string": "string1",
            "boolean": True,
            "int": 10,
            "long": 100,
            "float": 1000.0,
            "double": 10000.0,
            "repeated_bool": [True],
            "repeated_int": [20],
            "repeated_long": [200],
            "repeated_float": [1000.0],
            "repeated_double": [10000.0],
            "repeated_string": ["string1"],
        },
        {
            "string": "string2",
            "boolean": False,
            "int": 12,
            "long": 102,
            "float": 1002.0,
            "double": 10002.0,
            "repeated_bool": [True, False],
            "repeated_int": [20, 40],
            "repeated_long": [200, 400],
            "repeated_float": [1000.0, 800.0],
            "repeated_double": [101.0, 10.1],
            "repeated_string": ["string1", "string2"],
        },
    ]
    STREAM_2_ROWS = [
        {
            "string": "string2",
            "boolean": True,
            "int": 20,
            "long": 200,
            "float": 2000.0,
            "double": 20000.0,
            "repeated_bool": [True, False, True],
            "repeated_int": [20, 40, 30],
            "repeated_long": [200, 400, 700],
            "repeated_float": [1000.0, 800.0, 1100.0],
            "repeated_double": [101.0, 10.1, 0.3, 20.0],
            "repeated_string": ["string1", "string2", "string3"],
        },
        {
            # Empty record, all values are null except for repeated fields
            "repeated_bool": [False, True, True],
            "repeated_int": [30, 40, 20],
            "repeated_long": [200, 300, 900],
            "repeated_float": [1000.0, 700.0, 1200.0],
            "repeated_double": [101.0, 10.1, 0.3, 1.4],
            "repeated_string": ["string1", "string2", "string3", "string4"],
        },
    ]

    DEFAULT_VALUES = {
        "boolean": False,
        "double": 0.0,
        "float": 0.0,
        "int": 0,
        "long": 0,
        "string": "",
        "repeated_bool": [False, True, True],
        "repeated_int": [30, 40, 20],
        "repeated_long": [200, 300, 900],
        "repeated_float": [1000.0, 700.0, 1200.0],
        "repeated_double": [101.0, 10.1, 0.3, 1.4],
        "repeated_string": ["string1", "string2", "string3", "string4"],
        "repeated_string": ["string1", "string2", "string3", "string4"],
        "repeated_double": [101.0, 10.1, 0.3, 1.4],
    }

    SELECTED_FIELDS_LIST = [
        "string",
        "boolean",
        "int",
        "long",
        "float",
        "double",
    ]

    OUTPUT_TYPES_LIST = [
        dtypes.string,
        dtypes.bool,
        dtypes.int32,
        dtypes.int64,
        dtypes.float32,
        dtypes.float64,
    ]

    SELECTED_FIELDS_DICT = {
        "string": {"output_type": dtypes.string},
        "boolean": {"output_type": dtypes.bool},
        "int": {"output_type": dtypes.int32},
        "long": {"output_type": dtypes.int64},
        "float": {"output_type": dtypes.float32},
        "double": {"output_type": dtypes.float64},
        "repeated_bool": {
            "mode": BigQueryClient.FieldMode.REPEATED,
            "output_type": dtypes.bool,
        },
        "repeated_int": {
            "mode": BigQueryClient.FieldMode.REPEATED,
            "output_type": dtypes.int32,
        },
        "repeated_long": {
            "mode": BigQueryClient.FieldMode.REPEATED,
            "output_type": dtypes.int64,
        },
        "repeated_float": {
            "mode": BigQueryClient.FieldMode.REPEATED,
            "output_type": dtypes.float32,
        },
        "repeated_double": {
            "mode": BigQueryClient.FieldMode.REPEATED,
            "output_type": dtypes.float64,
        },
        "repeated_string": {
            "mode": BigQueryClient.FieldMode.REPEATED,
            "output_type": dtypes.string,
        },
    }

    SELECTED_FIELDS_DICT_WITH_DEFAULTS = {
        "string": {"output_type": dtypes.string, "default_value": "abc"},
        "boolean": {"output_type": dtypes.bool, "default_value": True},
        "int": {"output_type": dtypes.int32, "default_value": 10},
        "long": {"output_type": dtypes.int64, "default_value": 100},
        "float": {"output_type": dtypes.float32, "default_value": 100.0},
        "double": {"output_type": dtypes.float64, "default_value": 1000.0},
        "repeated_bool": {
            "mode": BigQueryClient.FieldMode.REPEATED,
            "output_type": dtypes.bool,
            "default_value": True,
        },
        "repeated_int": {
            "mode": BigQueryClient.FieldMode.REPEATED,
            "output_type": dtypes.int32,
            "default_value": -10,
        },
        "repeated_long": {
            "mode": BigQueryClient.FieldMode.REPEATED,
            "output_type": dtypes.int64,
            "default_value": -100,
        },
        "repeated_float": {
            "mode": BigQueryClient.FieldMode.REPEATED,
            "output_type": dtypes.float32,
            "default_value": -1000.01,
        },
        "repeated_double": {
            "mode": BigQueryClient.FieldMode.REPEATED,
            "output_type": dtypes.float64,
            "default_value": -1000.001,
        },
        "repeated_string": {
            "mode": BigQueryClient.FieldMode.REPEATED,
            "output_type": dtypes.string,
            "default_value": "def",
        },
    }

    CUSTOM_DEFAULT_VALUES = {
        "boolean": True,
        "double": 1000.0,
        "float": 100.0,
        "int": 10,
        "long": 100,
        "string": "abc",
        "repeated_bool": [False, True, True],
        "repeated_int": [30, 40, 20],
        "repeated_long": [200, 300, 900],
        "repeated_float": [1000.0, 700.0, 1200.0],
        "repeated_double": [101.0, 10.1, 0.3, 1.4],
        "repeated_string": ["string1", "string2", "string3", "string4"],
        "repeated_string": ["string1", "string2", "string3", "string4"],
        "repeated_double": [101.0, 10.1, 0.3, 1.4],
    }

    @staticmethod
    def _normalize_dictionary(dictionary):
        """Normalizes dictionary."""
        for key, value in dictionary.items():
            if isinstance(value, ops.Tensor):
                # for compartibility with Python 2
                dictionary[key] = value.numpy()
            value = dictionary[key]
            if isinstance(value, np.ndarray):
                lst = value.tolist()
                dictionary[key] = [
                    x.decode() if isinstance(x, bytes) else x for x in lst
                ]
            if isinstance(value, bytes):
                # because FakeBigQueryServer.serialize_to_avro serializes strings as byte arrays
                dictionary[key] = value.decode()
        return dict(dictionary)

    @staticmethod
    def _get_nonrepeated_only_fields(dictionary):
        nonrepeated_only_dict = {}
        for key, value in dictionary.items():
            if not key.startswith("repeated"):
                nonrepeated_only_dict[key] = value
        return nonrepeated_only_dict

    @classmethod
    def setUpClass(cls):  # pylint: disable=invalid-name
        """setUpClass"""
        cls.server = FakeBigQueryServer(
            cls.AVRO_SCHEMA, [cls.STREAM_1_ROWS, cls.STREAM_2_ROWS]
        )
        cls.server.start()

    @classmethod
    def tearDownClass(cls):  # pylint: disable=invalid-name
        """setUpClass"""
        cls.server.stop()

    def _get_read_session(
        self, client, selected_fields, output_types=None, requested_streams=2
    ):
        return client.read_session(
            self.PARENT,
            self.GCP_PROJECT_ID,
            self.TABLE_ID,
            self.DATASET_ID,
            selected_fields=selected_fields,
            output_types=output_types,
            requested_streams=2,
        )

    def test_fake_server(self):
        """Fake server test."""
        channel = grpc.insecure_channel(BigqueryOpsTest.server.endpoint())
        stub = storage_pb2_grpc.BigQueryStorageStub(channel)

        create_read_session_request = storage_pb2.CreateReadSessionRequest()
        create_read_session_request.table_reference.project_id = self.GCP_PROJECT_ID
        create_read_session_request.table_reference.dataset_id = self.DATASET_ID
        create_read_session_request.table_reference.table_id = self.TABLE_ID
        create_read_session_request.requested_streams = 2

        read_session_response = stub.CreateReadSession(create_read_session_request)
        self.assertEqual(2, len(read_session_response.streams))

        read_rows_request = storage_pb2.ReadRowsRequest()
        read_rows_request.read_position.stream.name = read_session_response.streams[
            0
        ].name
        read_rows_response = stub.ReadRows(read_rows_request)

        row = read_rows_response.next()
        self.assertEqual(
            FakeBigQueryServer.serialize_to_avro(self.STREAM_1_ROWS, self.AVRO_SCHEMA),
            row.avro_rows.serialized_binary_rows,
        )
        self.assertEqual(len(self.STREAM_1_ROWS), row.avro_rows.row_count)

        read_rows_request = storage_pb2.ReadRowsRequest()
        read_rows_request.read_position.stream.name = read_session_response.streams[
            1
        ].name
        read_rows_response = stub.ReadRows(read_rows_request)
        row = read_rows_response.next()
        self.assertEqual(
            FakeBigQueryServer.serialize_to_avro(self.STREAM_2_ROWS, self.AVRO_SCHEMA),
            row.avro_rows.serialized_binary_rows,
        )
        self.assertEqual(len(self.STREAM_2_ROWS), row.avro_rows.row_count)

    def test_read_rows(self):
        """Test for reading rows."""
        client = BigQueryTestClient(BigqueryOpsTest.server.endpoint())
        read_session = self._get_read_session(
            client, selected_fields=self.SELECTED_FIELDS_DICT
        )

        streams_list = read_session.get_streams()
        self.assertEqual(len(streams_list), 2)
        dataset1 = read_session.read_rows(streams_list[0])
        itr1 = iter(dataset1)
        self.assertEqual(
            self.STREAM_1_ROWS[0], self._normalize_dictionary(itr1.get_next())
        )
        self.assertEqual(
            self.STREAM_1_ROWS[1], self._normalize_dictionary(itr1.get_next())
        )
        with self.assertRaises(errors.OutOfRangeError):
            itr1.get_next()

        dataset2 = read_session.read_rows(streams_list[1])
        itr2 = iter(dataset2)
        self.assertEqual(
            self.STREAM_2_ROWS[0], self._normalize_dictionary(itr2.get_next())
        )
        self.assertEqual(
            self.DEFAULT_VALUES, self._normalize_dictionary(itr2.get_next())
        )
        with self.assertRaises(errors.OutOfRangeError):
            itr2.get_next()

    def test_read_rows_default_values(self):
        """Test for reading rows when default values are specified."""
        client = BigQueryTestClient(BigqueryOpsTest.server.endpoint())

        read_session = self._get_read_session(
            client, selected_fields=self.SELECTED_FIELDS_DICT_WITH_DEFAULTS
        )

        streams_list = read_session.get_streams()
        self.assertEqual(len(streams_list), 2)
        dataset2 = read_session.read_rows(streams_list[1])
        itr2 = iter(dataset2)
        self.assertEqual(
            self.STREAM_2_ROWS[0], self._normalize_dictionary(itr2.get_next())
        )
        self.assertEqual(
            self.CUSTOM_DEFAULT_VALUES, self._normalize_dictionary(itr2.get_next())
        )
        with self.assertRaises(errors.OutOfRangeError):
            itr2.get_next()

    def test_read_rows_nonrepeated_only(self):
        """Test for reading rows with non-repeated fields only, then selected_fields and output_types are list (backward compatible)."""
        client = BigQueryTestClient(BigqueryOpsTest.server.endpoint())
        read_session = self._get_read_session(
            client,
            selected_fields=self.SELECTED_FIELDS_LIST,
            output_types=self.OUTPUT_TYPES_LIST,
        )

        streams_list = read_session.get_streams()
        self.assertEqual(len(streams_list), 2)
        dataset1 = read_session.read_rows(streams_list[0])
        itr1 = iter(dataset1)
        self.assertEqual(
            self._get_nonrepeated_only_fields(self.STREAM_1_ROWS[0]),
            self._normalize_dictionary(itr1.get_next()),
        )
        self.assertEqual(
            self._get_nonrepeated_only_fields(self.STREAM_1_ROWS[1]),
            self._normalize_dictionary(itr1.get_next()),
        )
        with self.assertRaises(errors.OutOfRangeError):
            itr1.get_next()

        dataset2 = read_session.read_rows(streams_list[1])
        itr2 = iter(dataset2)
        self.assertEqual(
            self._get_nonrepeated_only_fields(self.STREAM_2_ROWS[0]),
            self._normalize_dictionary(itr2.get_next()),
        )
        self.assertEqual(
            self._get_nonrepeated_only_fields(self.DEFAULT_VALUES),
            self._normalize_dictionary(itr2.get_next()),
        )
        with self.assertRaises(errors.OutOfRangeError):
            itr2.get_next()

    def test_read_rows_with_offset(self):
        """Test for reading rows with offset."""
        client = BigQueryTestClient(BigqueryOpsTest.server.endpoint())
        read_session = self._get_read_session(
            client, selected_fields=self.SELECTED_FIELDS_DICT
        )

        streams_list = read_session.get_streams()
        self.assertEqual(len(streams_list), 2)
        dataset1 = read_session.read_rows(streams_list[0], offset=1)
        itr1 = iter(dataset1)
        self.assertEqual(
            self.STREAM_1_ROWS[1], self._normalize_dictionary(itr1.get_next())
        )
        with self.assertRaises(errors.OutOfRangeError):
            itr1.get_next()

    def test_parallel_read_rows(self):
        """Test for reading rows in parallel."""
        client = BigQueryTestClient(BigqueryOpsTest.server.endpoint())
        read_session = self._get_read_session(
            client, selected_fields=self.SELECTED_FIELDS_DICT
        )

        dataset = read_session.parallel_read_rows()
        itr = iter(dataset)
        self.assertEqual(
            self.STREAM_1_ROWS[0], self._normalize_dictionary(itr.get_next())
        )
        self.assertEqual(
            self.STREAM_2_ROWS[0], self._normalize_dictionary(itr.get_next())
        )
        self.assertEqual(
            self.STREAM_1_ROWS[1], self._normalize_dictionary(itr.get_next())
        )
        self.assertEqual(
            self.DEFAULT_VALUES, self._normalize_dictionary(itr.get_next())
        )


if __name__ == "__main__":
    test.main()
