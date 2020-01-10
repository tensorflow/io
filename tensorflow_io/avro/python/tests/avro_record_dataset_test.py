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

# Examples: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/data/experimental/kernel_tests/stats_dataset_test_base.py
from functools import reduce
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes as tf_types
from tensorflow_io.core.python.experimental.avro_record_dataset_ops import AvroRecordDataset
from tensorflow_io.avro.python.utils.avro_serialization import AvroFileToRecords, AvroSerializer

from tensorflow_io.avro.python.tests.avro_dataset_test_base import AvroDatasetTestBase


class AvroRecordDatasetTest(AvroDatasetTestBase):

    @staticmethod
    def _load_records_as_tensors(filenames, schema):
        serializer = AvroSerializer(schema)
        return map(lambda s: ops.convert_to_tensor(serializer.serialize(s), dtype=tf_types.string),
                   reduce(lambda a, b: a+b, [AvroFileToRecords(filename).get_records() for filename in filenames]))

    def _test_pass_dataset(self, writer_schema, record_data, **kwargs):
        filenames = AvroRecordDatasetTest._setup_files(writer_schema=writer_schema,
                                                       records=record_data)
        expected_data = AvroRecordDatasetTest._load_records_as_tensors(filenames, writer_schema)
        actual_dataset = AvroRecordDataset(
            filenames=filenames,
            num_parallel_reads=kwargs.get("num_parallel_reads", 1),
            reader_schema=kwargs.get("reader_schema"))

        data = iter(actual_dataset)
        for expected in expected_data:
            self.assertValuesEqual(expected=expected, actual=next(data))

    def _test_pass_dataset_resolved(self, writer_schema, reader_schema, record_data, **kwargs):
        filenames = AvroRecordDatasetTest._setup_files(writer_schema=writer_schema,
                                                       records=record_data)
        expected_data = AvroRecordDatasetTest._load_records_as_tensors(filenames, reader_schema)
        actual_dataset = AvroRecordDataset(
            filenames=filenames,
            num_parallel_reads=kwargs.get("num_parallel_reads", 1),
            reader_schema=reader_schema)

        data = iter(actual_dataset)
        for expected in expected_data:
            self.assertValuesEqual(expected=expected, actual=next(data))

    def test_wout_reader_schema(self):
        writer_schema = """{
              "type": "record",
              "name": "dataTypes",
              "fields": [
                  {
                     "name":"index",
                     "type":"int"
                  },
                  {
                     "name":"string_value",
                     "type":"string"
                  }
              ]}"""
        record_data = [
            {
                "index": 0,
                "string_value": ""
            },
            {
                "index": 1,
                "string_value": "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"
            },
            {
                "index": 2,
                "string_value": "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789"
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema, record_data=record_data)

    def test_with_schema_projection(self):
        writer_schema = """{
              "type": "record",
              "name": "dataTypes",
              "fields": [
                  {
                     "name":"index",
                     "type":"int"
                  },
                  {
                     "name":"string_value",
                     "type":"string"
                  }
              ]}"""
        # Test projection
        reader_schema = """{
              "type": "record",
              "name": "dataTypes",
              "fields": [
                  {
                     "name":"string_value",
                     "type":"string"
                  }
              ]}"""
        record_data = [
            {
                "index": 0,
                "string_value": ""
            },
            {
                "index": 1,
                "string_value": "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"
            },
            {
                "index": 2,
                "string_value": "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789"
            }
        ]
        self._test_pass_dataset_resolved(writer_schema=writer_schema,
                                         reader_schema=reader_schema,
                                         record_data=record_data)

    def test_schema_type_promotion(self):
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"},
                  {"name": "long_value", "type": "long"}
              ]}"""
        reader_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "long"},
                  {"name": "long_value", "type": "double"}
              ]}"""
        record_data = [
            {"int_value": 0, "long_value": 111},
            {"int_value": 1, "long_value": 222}
        ]
        self._test_pass_dataset_resolved(writer_schema=writer_schema,
                                         reader_schema=reader_schema,
                                         record_data=record_data)