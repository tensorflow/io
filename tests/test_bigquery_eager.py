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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import concurrent.futures
from io import BytesIO
import json
import fastavro
import grpc
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow import test
from tensorflow_io.bigquery import BigQueryTestClient

import google.cloud.bigquery.storage.v1beta1.storage_pb2_grpc as storage_pb2_grpc
import google.cloud.bigquery.storage.v1beta1.storage_pb2 as storage_pb2

if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()


class FakeBigQueryServer(storage_pb2_grpc.BigQueryStorageServicer):
  """Fake server for Cloud BigQuery Storage API."""

  def __init__(self, avro_schema, avro_serialized_rows_per_stream,
               avro_serialized_rows_count_per_stream):
    self._project_id = ""
    self._table_id = ""
    self._dataset_id = ""
    self._streams = list()
    self._avro_serialized_rows_per_stream = avro_serialized_rows_per_stream
    self._avro_serialized_rows_count_per_stream = \
        avro_serialized_rows_count_per_stream
    self._avro_schema = avro_schema

    self._grpc_server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=4))
    storage_pb2_grpc.add_BigQueryStorageServicer_to_server(
        self, self._grpc_server)
    port = self._grpc_server.add_insecure_port("localhost:0")
    self._endpoint = "localhost:" + str(port)
    print ("started server on :" + self._endpoint)

  def start(self):
    self._grpc_server.start()

  def stop(self):
    self._grpc_server.stop(0)

  def endpoint(self):
    return self._endpoint

  def _build_stream_name(self, stream_index):
    return self._project_id + "/" + self._dataset_id + "/" + \
        self._table_id + "/" + str(stream_index)

  def CreateReadSession(self, request, context):
    print ("called CreateReadSession on server")
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

  def ReadRows(self, request, context):
    print ("called ReadRows on server")
    response = storage_pb2.ReadRowsResponse()
    stream_index = self._streams.index(request.read_position.stream.name)
    if stream_index >= 0 and stream_index < len(
        self._avro_serialized_rows_per_stream):
      response.avro_rows.serialized_binary_rows = \
          self._avro_serialized_rows_per_stream[stream_index]
      response.avro_rows.row_count = \
          self._avro_serialized_rows_count_per_stream[stream_index]
    yield response


class BigqueryOpsTest(test.TestCase):
  """Tests for BigQuery adapter."""

  GCP_PROJECT_ID = "bigquery-public-data"
  DATASET_ID = "usa_names"
  TABLE_ID = "usa_1910_current"
  PARENT = "projects/some_parent"

  AVRO_SCHEMA = """
      {
      "type": "record",
      "name": "__root__",
      "fields": [
          {
              "name": "state",
              "type": [
                  "null",
                  "string"
              ],
              "doc": "2-digit state code"
          },
          {
              "name": "name",
              "type": [
                  "null",
                  "string"
              ],
              "doc": "Given name of a person at birth"
          },
          {
              "name": "number",
              "type": [
                  "null",
                  "long"
              ],
              "doc": "Number of occurrences of the name"
          }
      ]
  }"""

  STREAM_1_ROWS = [{
      "state": "wa",
      "name": "Andrew",
      "number": 1
  }, {
      "state": "wa",
      "name": "Eva",
      "number": 2
  }]
  STREAM_2_ROWS = [{"state": "ny", "name": "Emma", "number": 10}]

  @staticmethod
  def _serialize_to_avro(rows, schema):
    """Serializes specified rows into avro format."""
    string_output = BytesIO()
    json_schema = json.loads(schema)
    for row in rows:
      fastavro.schemaless_writer(string_output, json_schema, row)
    avro_output = string_output.getvalue()
    string_output.close()
    return avro_output

  @staticmethod
  def _normalize_dictionary(dictionary):
    """Normalizes dictionary."""
    for key, value in dictionary.items():
      if isinstance(value, ops.Tensor):
        # for compartibility with Python 2
        dictionary[key] = value.numpy()
      value = dictionary[key]
      if isinstance(value, bytes):
        # because _serialize_to_avro serialize strings as byte arrays
        dictionary[key] = value.decode()
    return dict(dictionary)

  @classmethod
  def setUpClass(cls): # pylint: disable=invalid-name
    """setUpClass"""
    cls.server = FakeBigQueryServer(cls.AVRO_SCHEMA, [
        cls._serialize_to_avro(cls.STREAM_1_ROWS, cls.AVRO_SCHEMA),
        cls._serialize_to_avro(cls.STREAM_2_ROWS, cls.AVRO_SCHEMA),
    ], [len(cls.STREAM_1_ROWS),
        len(cls.STREAM_2_ROWS)])
    cls.server.start()

  @classmethod
  def tearDownClass(cls): # pylint: disable=invalid-name
    """setUpClass"""
    cls.server.stop()

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
        0].name
    read_rows_response = stub.ReadRows(read_rows_request)

    row = read_rows_response.next()
    self.assertEqual(
        self._serialize_to_avro(self.STREAM_1_ROWS, self.AVRO_SCHEMA),
        row.avro_rows.serialized_binary_rows)
    self.assertEqual(len(self.STREAM_1_ROWS), row.avro_rows.row_count)

    read_rows_request = storage_pb2.ReadRowsRequest()
    read_rows_request.read_position.stream.name = read_session_response.streams[
        1].name
    read_rows_response = stub.ReadRows(read_rows_request)
    row = read_rows_response.next()
    self.assertEqual(
        self._serialize_to_avro(self.STREAM_2_ROWS, self.AVRO_SCHEMA),
        row.avro_rows.serialized_binary_rows)
    self.assertEqual(len(self.STREAM_2_ROWS), row.avro_rows.row_count)

  def test_read_rows(self):
    """Test for reading rows."""
    client = BigQueryTestClient(BigqueryOpsTest.server.endpoint())
    read_session = client.read_session(
        self.PARENT,
        self.GCP_PROJECT_ID,
        self.TABLE_ID,
        self.DATASET_ID, ["state", "name", "number"],
        [dtypes.string, dtypes.string, dtypes.int64],
        requested_streams=2)

    streams_list = read_session.get_streams()
    self.assertEqual(len(streams_list), 2)
    dataset1 = read_session.read_rows(streams_list[0])
    itr1 = iter(dataset1)
    self.assertEqual(self.STREAM_1_ROWS[0],
                     self._normalize_dictionary(itr1.get_next()))
    self.assertEqual(self.STREAM_1_ROWS[1],
                     self._normalize_dictionary(itr1.get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      itr1.get_next()

    dataset2 = read_session.read_rows(streams_list[1])
    itr2 = iter(dataset2)
    self.assertEqual(self.STREAM_2_ROWS[0],
                     self._normalize_dictionary(itr2.get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      itr2.get_next()

  def test_parallel_read_rows(self):
    """Test for reading rows in parallel."""
    client = BigQueryTestClient(BigqueryOpsTest.server.endpoint())
    read_session = client.read_session(
        self.PARENT,
        self.GCP_PROJECT_ID,
        self.TABLE_ID,
        self.DATASET_ID, ["state", "name", "number"],
        [dtypes.string, dtypes.string, dtypes.int64],
        requested_streams=2)

    dataset = read_session.parallel_read_rows()
    itr = iter(dataset)
    self.assertEqual(self.STREAM_1_ROWS[0],
                     self._normalize_dictionary(itr.get_next()))
    self.assertEqual(self.STREAM_2_ROWS[0],
                     self._normalize_dictionary(itr.get_next()))
    self.assertEqual(self.STREAM_1_ROWS[1],
                     self._normalize_dictionary(itr.get_next()))

if __name__ == "__main__":
  test.main()
