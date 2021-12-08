# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""AvroDatasetTest"""
# pylint: disable=line-too-long
# see https://github.com/tensorflow/io/pull/962#issuecomment-632346602

import sys
from functools import reduce
import os
import tempfile
from io import BytesIO
import pytest
import numpy as np

import tensorflow as tf
from avro.io import DatumReader, DatumWriter, BinaryDecoder, BinaryEncoder
from avro.datafile import DataFileReader, DataFileWriter
from avro.schema import Parse as parse
import tensorflow_io as tfio

if sys.platform == "darwin":
    pytest.skip("TODO: skip macOS", allow_module_level=True)


class AvroRecordsToFile:
    """AvroRecordsToFile"""

    def __init__(self, filename, writer_schema, codec="deflate"):
        """

        :param filename:
        :param writer_schema:
        :param codec:
        """
        self.schema = AvroParser(writer_schema).get_schema_object()
        self.filename = filename
        self.codec = codec

    def write_records(self, records):
        with open(self.filename, "wb") as out:
            writer = DataFileWriter(out, DatumWriter(), self.schema, codec=self.codec)
            for record in records:
                writer.append(record)
            writer.close()


class AvroFileToRecords:
    """AvroFileToRecords"""

    def __init__(self, filename, reader_schema=None):
        """
        Reads records as strings where each row is serialized separately

        :param filename: The filename from where to load the records
        :param reader_schema: Schema used for reading

        :return: An array of serialized string with one string per record
        """
        self.records = []

        with open(filename, "rb") as file_handle:
            datum_reader = (
                DatumReader(reader_schema=AvroParser(reader_schema).get_schema_object())
                if reader_schema
                else DatumReader()
            )
            reader = DataFileReader(file_handle, datum_reader)

            self.records += list(reader)

    def get_records(self):
        return self.records


class AvroSchemaReader:
    """AvroSchemaReader"""

    def __init__(self, filename):
        """
        Reads the schema from a file into json string
        """
        with open(filename, "rb") as file_handle:
            reader = DataFileReader(file_handle, DatumReader())
            self.schema_json = ""
            self.schema_json = str(reader.datum_reader.writer_schema)

    def get_schema_json(self):
        return self.schema_json


class AvroParser:
    """AvroParser"""

    def __init__(self, schema_json):
        """
        Create an avro parser mostly to abstract away the API change between
        avro and avro-python3

        :param schema_json:
        """
        self.schema_object = parse(schema_json)

    def get_schema_object(self):
        return self.schema_object


class AvroDeserializer:
    """AvroDeserializer"""

    def __init__(self, schema_json):
        """
        Create an avro deserializer.

        :param schema_json: Json string of the schema.
        """
        schema_object = AvroParser(schema_json).get_schema_object()
        # No schema resolution
        self.datum_reader = DatumReader(schema_object, schema_object)

    def deserialize(self, serialized_bytes):
        """
        Deserialize an avro record from bytes.

        :param serialized_bytes: The serialized bytes input.

        :return: The de-serialized record structure in python as map-list object.
        """
        return self.datum_reader.read(BinaryDecoder(BytesIO(serialized_bytes)))


class AvroSerializer:
    """AvroSerializer"""

    def __init__(self, schema_json):
        """
        Create an avro serializer.

        :param schema_json: Json string of the schema.
        """
        self.datum_writer = DatumWriter(AvroParser(schema_json).get_schema_object())

    def serialize(self, datum):
        """
        Serialize a datum into a avro formatted string.

        :param datum: The avro datum.

        :return: The serialized bytes.
        """
        writer = BytesIO()
        self.datum_writer.write(datum, BinaryEncoder(writer))
        return writer.getvalue()


class AvroDatasetTestBase(tf.test.TestCase):
    """AvroDatasetTestBase"""

    @staticmethod
    def _setup_files(writer_schema, records):
        """setup_files"""
        # Write test records into temporary output directory
        filename = os.path.join(tempfile.mkdtemp(), "test.avro")
        writer = AvroRecordsToFile(filename=filename, writer_schema=writer_schema)
        writer.write_records(records)

        return [filename]

    def assert_values_equal(self, expected, actual):
        """Asserts that two values are equal."""
        if isinstance(expected, dict):
            self.assertItemsEqual(list(expected.keys()), list(actual.keys()))
            for k in expected.keys():
                self.assert_values_equal(expected[k], actual[k])
        elif isinstance(expected, (tf.SparseTensor, tf.compat.v1.SparseTensorValue)):
            self.assertAllEqual(expected.indices, actual.indices)
            self.assertAllEqual(expected.values, actual.values)
            self.assertAllEqual(expected.dense_shape, actual.dense_shape)
        else:
            self.assertAllEqual(expected, actual)

    def assert_data_equal(self, expected, actual):
        """assert_data_equal"""

        def _assert_equal(expected, actual):
            for name, datum in expected.items():
                self.assert_values_equal(expected=datum, actual=actual[name])

        if isinstance(expected, tuple):
            assert isinstance(
                expected, tuple
            ), f"Found type {type(actual)} but expected type {tuple}"
            assert (
                len(expected) == 2
            ), "Found {} components in expected dataset but must have {}".format(
                len(expected), 2
            )

            assert (
                len(actual) == 2
            ), "Found {} components in actual dataset but expected {}".format(
                len(actual), 2
            )

            expected_features, expected_labels = expected
            actual_features, actual_labels = actual

            _assertEqual(expected_features, actual_features)
            _assertEqual(expected_labels, actual_labels)

        else:
            _assert_equal(expected, actual)

    def _verify_output(self, expected_data, actual_dataset):

        next_data = iter(actual_dataset)

        for expected in expected_data:
            self.assert_data_equal(expected=expected, actual=next(next_data))


class AvroRecordDatasetTest(AvroDatasetTestBase):
    """AvroRecordDatasetTest"""

    @staticmethod
    def _load_records_as_tensors(filenames, schema):
        serializer = AvroSerializer(schema)
        return map(
            lambda s: tf.convert_to_tensor(
                serializer.serialize(s), dtype=tf.dtypes.string
            ),
            reduce(
                lambda a, b: a + b,
                [AvroFileToRecords(filename).get_records() for filename in filenames],
            ),
        )

    def test_inval_num_parallel_calls(self):
        """test_inval_num_parallel_calls
        This function tests that value errors are raised upon
        the passing of invalid values for num_parallel_calls which
        includes zero values and values greater than num_parallel_reads
        """

        NUM_PARALLEL_READS = 1
        NUM_PARALLEL_CALLS_ZERO = 0
        NUM_PARALLEL_CALLS_GREATER = 2

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
            {"index": 0, "string_value": ""},
            {"index": 1, "string_value": "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\'?"},
            {
                "index": 2,
                "string_value": "ABCDEFGHIJKLMNOPQRSTUVW"
                + "Zabcdefghijklmnopqrstuvwz0123456789",
            },
        ]

        filenames = AvroRecordDatasetTest._setup_files(
            writer_schema=writer_schema, records=record_data
        )

        with pytest.raises(ValueError):

            dataset_a = tfio.experimental.columnar.AvroRecordDataset(
                filenames=filenames,
                num_parallel_reads=NUM_PARALLEL_READS,
                num_parallel_calls=NUM_PARALLEL_CALLS_ZERO,
                reader_schema="reader_schema",
            )

        with pytest.raises(ValueError):

            dataset_b = tfio.experimental.columnar.AvroRecordDataset(
                filenames=filenames,
                num_parallel_reads=NUM_PARALLEL_READS,
                num_parallel_calls=NUM_PARALLEL_CALLS_GREATER,
                reader_schema="reader_schema",
            )

    def _test_pass_dataset(self, writer_schema, record_data, **kwargs):
        """test_pass_dataset"""
        filenames = AvroRecordDatasetTest._setup_files(
            writer_schema=writer_schema, records=record_data
        )
        expected_data = AvroRecordDatasetTest._load_records_as_tensors(
            filenames, writer_schema
        )
        actual_dataset = tfio.experimental.columnar.AvroRecordDataset(
            filenames=filenames,
            num_parallel_reads=kwargs.get("num_parallel_reads", 1),
            reader_schema=kwargs.get("reader_schema"),
        )
        data = iter(actual_dataset)
        for expected in expected_data:
            self.assert_values_equal(expected=expected, actual=next(data))

    def _test_pass_dataset_resolved(
        self, writer_schema, reader_schema, record_data, **kwargs
    ):
        """test_pass_dataset_resolved"""
        filenames = AvroRecordDatasetTest._setup_files(
            writer_schema=writer_schema, records=record_data
        )
        expected_data = AvroRecordDatasetTest._load_records_as_tensors(
            filenames, reader_schema
        )
        actual_dataset = tfio.experimental.columnar.AvroRecordDataset(
            filenames=filenames,
            num_parallel_reads=kwargs.get("num_parallel_reads", 1),
            reader_schema=reader_schema,
        )

        data = iter(actual_dataset)
        for expected in expected_data:
            self.assert_values_equal(expected=expected, actual=next(data))

    def test_wout_reader_schema(self):
        """test_wout_reader_schema"""
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
            {"index": 0, "string_value": ""},
            {"index": 1, "string_value": "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\'?"},
            {
                "index": 2,
                "string_value": "ABCDEFGHIJKLMNOPQRSTUVW"
                + "Zabcdefghijklmnopqrstuvwz0123456789",
            },
        ]
        self._test_pass_dataset(writer_schema=writer_schema, record_data=record_data)

    @pytest.mark.skip(reason="failed with tf 2.2 rc3 on linux")
    def test_with_schema_projection(self):
        """test_with_schema_projection"""
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
            {"index": 0, "string_value": ""},
            {"index": 1, "string_value": "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\'?"},
            {
                "index": 2,
                "string_value": "ABCDEFGHIJKLMNOPQRSTUVWZabcde"
                + "fghijklmnopqrstuvwz0123456789",
            },
        ]
        self._test_pass_dataset_resolved(
            writer_schema=writer_schema,
            reader_schema=reader_schema,
            record_data=record_data,
        )

    def test_schema_type_promotion(self):
        """test_schema_type_promotion"""
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
            {"int_value": 1, "long_value": 222},
        ]
        self._test_pass_dataset_resolved(
            writer_schema=writer_schema,
            reader_schema=reader_schema,
            record_data=record_data,
        )


class MakeAvroRecordDatasetTest(AvroDatasetTestBase):
    """MakeAvroRecordDatasetTest"""

    def _test_pass_dataset(
        self,
        writer_schema,
        record_data,
        expected_data,
        features,
        reader_schema,
        batch_size,
        **kwargs,
    ):
        """_test_pass_dataset"""
        filenames = AvroDatasetTestBase._setup_files(
            writer_schema=writer_schema, records=record_data
        )

        actual_dataset = tfio.experimental.columnar.make_avro_record_dataset(
            file_pattern=filenames,
            features=features,
            batch_size=batch_size,
            reader_schema=reader_schema,
            shuffle=kwargs.get("shuffle", None),
            num_epochs=kwargs.get("num_epochs", None),
        )

        self._verify_output(expected_data=expected_data, actual_dataset=actual_dataset)

    def test_variable_length_failed_with_wrong_rank(self):
        """test_variable_length_failed_with_wrong_rank"""
        reader_schema = """{
                    "type": "record",
                    "name": "row",
                    "fields": [
                        {
                           "name": "int_list_list",
                           "type": {
                              "type": "array",
                              "items": {
                                  "type": "array",
                                  "items": "int"
                              }
                           }
                        }
                    ]}"""
        record_data = [
            {"int_list_list": [[1, 2], [3, 4, 5]]},
            {"int_list_list": [[6]]},
            {"int_list_list": [[6]]},
        ]
        features = {
            "int_list_list[*][*]": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.int32
            )
        }
        expected_data = [
            {
                "int_list_list[*][*]": tf.compat.v1.SparseTensorValue(
                    indices=[
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [0, 1, 2],
                        [1, 0, 0],
                        [2, 0, 0],
                    ],
                    values=[1, 2, 3, 4, 5, 6, 6],
                    dense_shape=[3, 2, 3],
                )
            }
        ]
        with self.assertRaises(Exception) as context:
            self._test_pass_dataset(
                reader_schema=reader_schema,
                record_data=record_data,
                expected_data=expected_data,
                features=features,
                writer_schema=reader_schema,
                batch_size=3,
                num_epochs=1,
            )
            self.assertTrue(
                "is not compatible with supplied shape" in context.exception
            )

    def test_variable_length_passed_with_rank(self):
        """test_variable_length_passed_with_rank"""
        reader_schema = """{
                    "type": "record",
                    "name": "row",
                    "fields": [
                        {
                           "name": "int_list_list",
                           "type": {
                              "type": "array",
                              "items": {
                                  "type": "array",
                                  "items": "int"
                              }
                           }
                        }
                    ]}"""
        record_data = [
            {"int_list_list": [[1, 2], [3, 4, 5]]},
            {"int_list_list": [[6]]},
            {"int_list_list": [[6]]},
        ]
        features = {
            "int_list_list[*][*]": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.int32, 2
            )
        }
        expected_data = [
            {
                "int_list_list[*][*]": tf.compat.v1.SparseTensorValue(
                    indices=[
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [0, 1, 2],
                        [1, 0, 0],
                        [2, 0, 0],
                    ],
                    values=[1, 2, 3, 4, 5, 6, 6],
                    dense_shape=[3, 2, 3],
                )
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            writer_schema=reader_schema,
            batch_size=3,
            num_epochs=1,
        )

    def test_batching(self):
        """test_batching"""
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"}
              ]}"""
        record_data = [{"int_value": 0}, {"int_value": 1}, {"int_value": 2}]
        features = {"int_value": tf.io.FixedLenFeature([], tf.dtypes.int32)}
        expected_data = [
            {"int_value": tf.convert_to_tensor([0, 1])},
            {"int_value": tf.convert_to_tensor([2])},
        ]
        self._test_pass_dataset(
            writer_schema=writer_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            reader_schema=writer_schema,
            batch_size=2,
            num_epochs=1,
        )

    def test_fixed_length_list(self):
        """test_fixed_length_list"""
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_list",
                     "type": {
                        "type": "array",
                        "items": "int"
                     }
                  }
              ]}"""
        record_data = [
            {"int_list": [0, 1, 2]},
            {"int_list": [3, 4, 5]},
            {"int_list": [6, 7, 8]},
        ]
        features = {"int_list[*]": tf.io.FixedLenFeature([3], tf.dtypes.int32)}
        expected_data = [
            {"int_list[*]": tf.convert_to_tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])}
        ]

        self._test_pass_dataset(
            writer_schema=writer_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            reader_schema=writer_schema,
            batch_size=3,
            num_epochs=1,
        )


class ParseAvroDatasetTest(AvroDatasetTestBase):
    """AvroDatasetTest"""

    def assert_data_equal(self, expected, actual):
        """assert_data_equal"""
        for name, datum in expected.items():
            self.assert_values_equal(expected=datum, actual=actual[name])

    @staticmethod
    def _batcher(iterable, step):
        n = len(iterable)
        for ndx in range(0, n, step):
            yield iterable[ndx : min(ndx + step, n)]

    def _test_pass_dataset(
        self, reader_schema, record_data, expected_data, features, batch_size
    ):
        """_test_pass_dataset"""
        # Note, The batch size could be inferred from the expected data but found it better to be
        # explicit here
        serializer = AvroSerializer(reader_schema)
        for expected_datum, actual_records in zip(
            expected_data, ParseAvroDatasetTest._batcher(record_data, batch_size)
        ):
            # Get any key out of expected datum
            actual_datum = tfio.experimental.columnar.parse_avro(
                serialized=[
                    tf.convert_to_tensor(serializer.serialize(r))
                    for r in actual_records
                ],
                reader_schema=reader_schema,
                features=features,
            )
            self.assert_data_equal(expected=expected_datum, actual=actual_datum)

    def _test_fail_dataset(
        self, reader_schema, record_data, features, batch_size, **kwargs
    ):
        parser_schema = kwargs.get("parser_schema", reader_schema)
        serializer = AvroSerializer(reader_schema)
        for actual_records in ParseAvroDatasetTest._batcher(record_data, batch_size):
            # Get any key out of expected datum
            with self.assertRaises(tf.errors.OpError):
                _ = tfio.experimental.columnar.parse_avro(
                    serialized=[
                        tf.convert_to_tensor(serializer.serialize(r))
                        for r in actual_records
                    ],
                    reader_schema=parser_schema,
                    features=features,
                )

    @pytest.mark.skip(reason="failed with tf 2.2 rc3 on linux")
    def test_primitive_types(self):
        """test_primitive_types"""
        reader_schema = """{
              "type": "record",
              "name": "dataTypes",
              "fields": [
                  {
                     "name":"string_value",
                     "type":"string"
                  },
                  {
                     "name":"bytes_value",
                     "type":"bytes"
                  },
                  {
                     "name":"double_value",
                     "type":"double"
                  },
                  {
                     "name":"float_value",
                     "type":"float"
                  },
                  {
                     "name":"long_value",
                     "type":"long"
                  },
                  {
                     "name":"int_value",
                     "type":"int"
                  },
                  {
                     "name":"boolean_value",
                     "type":"boolean"
                  }
              ]}"""
        record_data = [
            {
                "string_value": "",
                "bytes_value": b"",
                "double_value": 0.0,
                "float_value": 0.0,
                "long_value": 0,
                "int_value": 0,
                "boolean_value": False,
            },
            {
                "string_value": "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\'?",
                "bytes_value": b"SpecialChars@!#$%^&*()-_=+{}[]|/`~\\'?",
                "double_value": -1.0,
                "float_value": -1.0,
                "long_value": 9223372036854775807,
                "int_value": 2147483648 - 1,
                "boolean_value": True,
            },
            {
                "string_value": "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghi"
                + "jklmnopqrstuvwz0123456789",
                "bytes_value": b"ABCDEFGHIJKLMNOPQRSTUVWZab"
                + "cdefghijklmnopqrstuvwz0123456789",
                "double_value": 1.0,
                "float_value": 1.0,
                "long_value": -9223372036854775807 - 1,
                "int_value": -2147483648,
                "boolean_value": False,
            },
        ]
        features = {
            "string_value": tf.io.FixedLenFeature([], tf.dtypes.string),
            "bytes_value": tf.io.FixedLenFeature([], tf.dtypes.string),
            "double_value": tf.io.FixedLenFeature([], tf.dtypes.float64),
            "float_value": tf.io.FixedLenFeature([], tf.dtypes.float32),
            "long_value": tf.io.FixedLenFeature([], tf.dtypes.int64),
            "int_value": tf.io.FixedLenFeature([], tf.dtypes.int32),
            "boolean_value": tf.io.FixedLenFeature([], tf.dtypes.bool),
        }
        expected_data = [
            {
                "string_value": tf.convert_to_tensor(
                    [
                        tf.compat.as_bytes(""),
                        tf.compat.as_bytes("SpecialChars@!#$%^&*()-_=+{}[]|/`~\\'?"),
                        tf.compat.as_bytes(
                            "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789"
                        ),
                    ]
                ),
                "bytes_value": tf.convert_to_tensor(
                    [
                        tf.compat.as_bytes(""),
                        tf.compat.as_bytes("SpecialChars@!#$%^&*()-_=+{}[]|/`~\\'?"),
                        tf.compat.as_bytes(
                            "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789"
                        ),
                    ]
                ),
                # Note, conversion utils `ops.EagerTensor` only seems to support single precision.
                # Proper values for double precision are 1.7976931348623157e+308, -1.7976931348623157e+308
                # In addition, precision is not maintained by the conversion, thus, I simplify set 1.0
                # and -1.0 instead of proper values 3.40282306074e+38 and -3.40282306074e+38.
                "double_value": tf.convert_to_tensor([0.0, -1.0, 1.0]),
                "float_value": tf.convert_to_tensor([0.0, -1.0, 1.0]),
                "long_value": tf.convert_to_tensor(
                    [0, 9223372036854775807, -9223372036854775807 - 1]
                ),
                "int_value": tf.convert_to_tensor([0, 2147483648 - 1, -2147483648]),
                "boolean_value": tf.convert_to_tensor([False, True, False]),
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=3,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_fixed_enum_types(self):
        """test_fixed_enum_types"""
        reader_schema = """{
              "type": "record",
              "name": "dataTypes",
              "fields": [
                  {
                     "name":"fixed_value",
                     "type": {
                        "name": "TenBytes",
                        "type": "fixed",
                        "size": 10
                     }
                  },
                  {
                     "name":"enum_value",
                     "type":{
                        "name": "Color",
                        "type": "enum",
                        "symbols": ["BLUE", "GREEN", "BROWN"]
                     }
                  }
              ]}"""
        record_data = [
            {"fixed_value": b"0123456789", "enum_value": "BLUE"},
            {"fixed_value": b"1234567890", "enum_value": "GREEN"},
            {"fixed_value": b"2345678901", "enum_value": "BROWN"},
        ]
        features = {
            "fixed_value": tf.io.FixedLenFeature([], tf.dtypes.string),
            "enum_value": tf.io.FixedLenFeature([], tf.dtypes.string),
        }
        expected_data = [
            {
                "fixed_value": tf.convert_to_tensor(
                    [
                        tf.compat.as_bytes("0123456789"),
                        tf.compat.as_bytes("1234567890"),
                        tf.compat.as_bytes("2345678901"),
                    ]
                ),
                "enum_value": tf.convert_to_tensor([b"BLUE", b"GREEN", b"BROWN"]),
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=3,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_batching(self):
        """test_batching"""
        reader_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"}
              ]}"""
        record_data = [{"int_value": 0}, {"int_value": 1}, {"int_value": 2}]
        features = {"int_value": tf.io.FixedLenFeature([], tf.dtypes.int32)}
        expected_data = [
            {"int_value": tf.convert_to_tensor([0, 1])},
            {"int_value": tf.convert_to_tensor([2])},
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_padding_from_default(self):
        """test_padding_from_default"""
        reader_schema = """{
                  "type": "record",
                  "name": "row",
                  "fields": [
                      {
                         "name": "fixed_len",
                         "type": {
                            "type": "array",
                            "items": "int"
                         }
                      }
                  ]}"""
        record_data = [
            {"fixed_len": [0]},
            {"fixed_len": [1]},
            {"fixed_len": [2]},
            {"fixed_len": [3]},
        ]
        features = {
            "fixed_len[*]": tf.io.FixedLenFeature(
                [2], tf.dtypes.int32, default_value=[0, 1]
            )
        }
        # Note, last batch is NOT dropped
        expected_data = [
            {"fixed_len[*]": tf.convert_to_tensor([[0, 1], [1, 1], [2, 1]])},
            {"fixed_len[*]": tf.convert_to_tensor([[3, 1]])},
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=3,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_batching_with_default(self):
        """test_batching_with_default"""
        reader_schema = """{
                  "type": "record",
                  "name": "row",
                  "fields": [
                      {
                         "name": "fixed_len",
                         "type": {
                            "type": "array",
                            "items": "int"
                         }
                      }
                  ]}"""
        record_data = [
            {"fixed_len": [0, 1, 2]},
            {"fixed_len": [3, 4, 5]},
            {"fixed_len": [6, 7, 8]},
        ]
        features = {
            "fixed_len[*]": tf.io.FixedLenFeature(
                [None, 3], tf.dtypes.int32, default_value=[0, 1, 2]
            )
        }
        expected_data = [
            {"fixed_len[*]": tf.convert_to_tensor([[0, 1, 2], [3, 4, 5]])},
            {"fixed_len[*]": tf.convert_to_tensor([[6, 7, 8]])},
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )

    def test_union_with_null(self):
        reader_schema = """{
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "possible_float_type",
                   "type": [
                      "null",
                      "float"
                   ]
                }
             ]
          }
          """
        record_data = [
            {"possible_float_type": 1.0},
            {"possible_float_type": None},
            {"possible_float_type": -1.0},
        ]
        features = {
            "possible_float_type:float": tf.io.FixedLenFeature(
                [], tf.dtypes.float32, default_value=0.0
            )
        }
        # If we have a default, then we use that in the place of the None
        expected_data = [
            {"possible_float_type:float": tf.convert_to_tensor([1.0, 0.0, -1.0])}
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=3,
        )

    def test_null_union_primitive_type(self):
        reader_schema = """{
             "type":"record",
             "name":"data_row",
             "fields":[
                {
                   "name":"multi_type",
                   "type":[
                      "null",
                      "boolean",
                      "int",
                      "long",
                      "float",
                      "double",
                      "string"
                   ]
                }
             ]
          }
          """
        record_data = [
            {"multi_type": None},
            {"multi_type": True},  # written as double(1.0)
            {"multi_type": int(1)},  # written as double(1.0)
            {"multi_type": 2},  # written as double(2.0)
            {"multi_type": float(3.0)},  # written as double(3.0)
            {"multi_type": 4.0},  # written as double (4.0)
            {"multi_type": "abc"},
        ]
        features = {
            "multi_type:boolean": tf.io.FixedLenFeature(
                [], tf.dtypes.bool, default_value=False
            ),
            "multi_type:int": tf.io.FixedLenFeature(
                [], tf.dtypes.int32, default_value=int(0)
            ),
            "multi_type:long": tf.io.FixedLenFeature(
                [], tf.dtypes.int64, default_value=0
            ),
            "multi_type:float": tf.io.FixedLenFeature(
                [], tf.dtypes.float32, default_value=float(0.0)
            ),
            "multi_type:double": tf.io.FixedLenFeature(
                [], tf.dtypes.float64, default_value=0.0
            ),
            "multi_type:string": tf.io.FixedLenFeature(
                [], tf.dtypes.string, default_value=""
            ),
        }
        expected_data = [
            {
                "multi_type:boolean": tf.convert_to_tensor(
                    [False, False, False, False, False, False, False],
                    dtype=tf.dtypes.bool,
                ),
                "multi_type:int": tf.convert_to_tensor(
                    [0, 0, 0, 0, 0, 0, 0], dtype=tf.dtypes.int32
                ),
                "multi_type:long": tf.convert_to_tensor(
                    [0, 0, 0, 0, 0, 0, 0], dtype=tf.dtypes.int64
                ),
                "multi_type:float": tf.convert_to_tensor(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.dtypes.float32
                ),
                "multi_type:double": tf.convert_to_tensor(
                    [0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 0.0], dtype=tf.dtypes.float64
                ),
                "multi_type:string": tf.convert_to_tensor(
                    [
                        tf.compat.as_bytes(""),
                        tf.compat.as_bytes(""),
                        tf.compat.as_bytes(""),
                        tf.compat.as_bytes(""),
                        tf.compat.as_bytes(""),
                        tf.compat.as_bytes(""),
                        tf.compat.as_bytes("abc"),
                    ]
                ),
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=7,
        )

    def test_union_without_default(self):
        reader_schema = """{
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "possible_float_type",
                   "type": [
                      "null",
                      "float"
                   ]
                }
             ]
          }
          """
        record_data = [{"possible_float_type": None}]
        features = {
            "possible_float_type:float": tf.io.FixedLenFeature([], tf.dtypes.float32)
        }
        self._test_fail_dataset(reader_schema, record_data, features, 1)

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_fixed_length_list(self):
        """test_fixed_length_list"""
        reader_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_list",
                     "type": {
                        "type": "array",
                        "items": "int"
                     }
                  }
              ]}"""
        record_data = [
            {"int_list": [0, 1, 2]},
            {"int_list": [3, 4, 5]},
            {"int_list": [6, 7, 8]},
        ]
        features = {"int_list[*]": tf.io.FixedLenFeature([3], tf.dtypes.int32)}
        expected_data = [
            {"int_list[*]": tf.convert_to_tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])}
        ]

        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=3,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_fixed_length_with_default_vector(self):
        """test_fixed_length_with_default_vector"""
        reader_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_list",
                     "type": {
                        "type": "array",
                        "items": "int"
                     }
                  }
              ]}"""
        record_data = [{"int_list": [0, 1, 2]}, {"int_list": [3]}, {"int_list": [6, 7]}]
        features = {
            "int_list[*]": tf.io.FixedLenFeature(
                [None, 3], tf.dtypes.int32, default_value=[0, 1, 2]
            )
        }
        expected_data = [
            {"int_list[*]": tf.convert_to_tensor([[0, 1, 2], [3, 1, 2], [6, 7, 2]])}
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=3,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_fixed_length_with_default_scalar(self):
        """test_fixed_length_with_default_scalar"""
        reader_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_list",
                     "type": {
                        "type": "array",
                        "items": "int"
                     }
                  }
              ]}"""
        record_data = [{"int_list": [0, 1, 2]}, {"int_list": [3]}, {"int_list": [6, 7]}]
        features = {
            "int_list[*]": tf.io.FixedLenFeature(
                [None], tf.dtypes.int32, default_value=0
            )
        }
        expected_data = [
            {"int_list[*]": tf.convert_to_tensor([[0, 1, 2], [3, 0, 0], [6, 7, 0]])}
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=3,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_dense_2d(self):
        """test_dense_2d"""
        reader_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_list",
                     "type": {
                        "type": "array",
                        "items":
                          {
                             "name" : "name",
                             "type" : "record",
                             "fields" : [
                                {
                                   "name": "nested_int_list",
                                   "type":
                                      {
                                          "type": "array",
                                          "items": "int"
                                      }
                                }
                             ]
                          }
                     }
                  }
              ]}"""
        record_data = [
            {
                "int_list": [
                    {"nested_int_list": [1, 2, 3]},
                    {"nested_int_list": [4, 5, 6]},
                ]
            },
            {
                "int_list": [
                    {"nested_int_list": [7, 8, 9]},
                    {"nested_int_list": [10, 11, 12]},
                ]
            },
        ]
        features = {
            "int_list[*].nested_int_list[*]": tf.io.FixedLenFeature(
                [2, 3], tf.dtypes.int32
            )
        }
        expected_data = [
            {
                "int_list[*].nested_int_list[*]": tf.convert_to_tensor(
                    [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
                )
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_dense_array_3d(self):
        """test_dense_array_3d"""
        # Here we use arrays directly for the nesting
        reader_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_list",
                     "type": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": "int"
                        }
                     }
                  }
              ]
              }"""
        record_data = [
            {"int_list": [[0, 1, 2], [10, 11, 12], [20, 21, 22]]},
            {"int_list": [[1, 2, 3], [11, 12, 13], [21, 22, 23]]},
        ]
        # Note, need to at least define the rank of the data, dimension can be unknown
        # This is a limitation inside TensorFlow where shape ranks need to be known
        # inside _from_compatible_tensor_list
        features = {
            "int_list[*][*]": tf.io.FixedLenFeature([None, None], tf.dtypes.int32)
        }
        # Note, the outer dimension is the batch dimension
        expected_data = [
            {
                "int_list[*][*]": tf.convert_to_tensor(
                    [
                        [[0, 1, 2], [10, 11, 12], [20, 21, 22]],
                        [[1, 2, 3], [11, 12, 13], [21, 22, 23]],
                    ]
                )
            },
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )

    @pytest.mark.skip(reason="failed with tf 2.2 rc3 on linux")
    def test_sparse_feature(self):
        """test_sparse_feature"""
        reader_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                {
                  "name": "sparse_type",
                  "type": {
                    "type": "array",
                    "items": {
                       "type": "record",
                       "name": "sparse_triplet",
                       "fields": [
                          {
                             "name":"index",
                             "type":"long"
                          },
                          {
                             "name":"value",
                             "type":"float"
                          }
                       ]
                    }
                 }
              }
        ]}"""
        record_data = [
            {"sparse_type": [{"index": 0, "value": 5.0}, {"index": 3, "value": 2.0}]},
            {"sparse_type": [{"index": 2, "value": 7.0}]},
            {"sparse_type": [{"index": 1, "value": 6.0}]},
            {"sparse_type": [{"index": 3, "value": 3.0}]},
        ]
        features = {
            "sparse_type": tf.io.SparseFeature(
                index_key="index", value_key="value", dtype=tf.dtypes.float32, size=4
            )
        }
        expected_data = [
            {
                "sparse_type": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [0, 3], [1, 2]],
                    values=[5.0, 2.0, 7.0],
                    dense_shape=[2, 4],
                )
            },
            {
                "sparse_type": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 1], [1, 3]], values=[6.0, 3.0], dense_shape=[2, 4]
                )
            },
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )

    @pytest.mark.skip(reason="failed with tf 2.2 rc3 on linux")
    def test_type_reuse(self):
        """test_type_reuse"""
        reader_schema = """{
            "type": "record",
            "name": "row",
            "fields": [
              {
                "name": "first_value",
                "type": {
                  "type": "array",
                  "items": {
                     "type": "record",
                     "name": "Tuple",
                     "fields": [
                        {
                           "name":"index",
                           "type":"long"
                        },
                        {
                           "name":"value",
                           "type":"float"
                        }
                     ]
                  }
              }
            },
            {
              "name": "second_value",
              "type": {
                "type": "array",
                "items": "Tuple"
              }
            }
          ]
          }"""
        record_data = [
            {
                "first_value": [{"index": 0, "value": 5.0}, {"index": 3, "value": 2.0}],
                "second_value": [{"index": 2, "value": 7.0}],
            },
            {
                "first_value": [{"index": 0, "value": 2.0}],
                "second_value": [{"index": 1, "value": 2.0}],
            },
        ]
        features = {
            "first_value": tf.io.SparseFeature(
                index_key="index", value_key="value", dtype=tf.dtypes.float32, size=4
            ),
            "second_value": tf.io.SparseFeature(
                index_key="index", value_key="value", dtype=tf.dtypes.float32, size=3
            ),
        }
        expected_data = [
            {
                "first_value": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [0, 3], [1, 0]],
                    values=[5.0, 2.0, 2.0],
                    dense_shape=[2, 4],
                ),
                "second_value": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 2], [1, 1]], values=[7.0, 2.0], dense_shape=[2, 3]
                ),
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_variable_length(self):
        """test_variable_length"""
        reader_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_list",
                     "type": {
                        "type": "array",
                        "items": "int"
                     }
                  }
              ]}"""
        record_data = [{"int_list": [1, 2]}, {"int_list": [3, 4, 5]}, {"int_list": [6]}]
        features = {
            "int_list[*]": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.int32, 1
            )
        }
        expected_data = [
            {
                "int_list[*]": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0]],
                    values=[1, 2, 3, 4, 5, 6],
                    dense_shape=[3, 3],
                )
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=3,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_variable_length_2d(self):
        """test_variable_length_2d"""
        reader_schema = """{
                  "type": "record",
                  "name": "row",
                  "fields": [
                      {
                         "name": "int_list_list",
                         "type": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": "int"
                            }
                         }
                      }
                  ]}"""
        record_data = [
            {"int_list_list": [[1, 2], [3, 4, 5]]},
            {"int_list_list": [[6]]},
            {"int_list_list": [[6]]},
        ]
        features = {
            "int_list_list[*][*]": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.int32, 2
            )
        }
        expected_data = [
            {
                "int_list_list[*][*]": tf.compat.v1.SparseTensorValue(
                    indices=[
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [0, 1, 2],
                        [1, 0, 0],
                        [2, 0, 0],
                    ],
                    values=[1, 2, 3, 4, 5, 6, 6],
                    dense_shape=[3, 2, 3],
                )
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=3,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_nesting(self):
        """test_nesting"""
        reader_schema = """{
           "type": "record",
           "name": "nesting",
           "fields": [
              {
                 "name": "nested_record",
                 "type": {
                    "type": "record",
                    "name": "nested_values",
                    "fields": [
                       {
                          "name": "nested_int",
                          "type": "int"
                       },
                       {
                          "name": "nested_float_list",
                          "type": {
                             "type": "array",
                             "items": "float"
                          }
                       }
                    ]
                 }
              },
              {
                 "name": "list_of_records",
                 "type": {
                    "type": "array",
                    "items": {
                       "type": "record",
                       "name": "person",
                       "fields": [
                          {
                             "name": "first_name",
                             "type": "string"
                          },
                          {
                             "name": "age",
                             "type": "int"
                          }
                       ]
                    }
                 }
              }
           ]
        }
        """
        record_data = [
            {
                "nested_record": {"nested_int": 0, "nested_float_list": [0.0, 10.0]},
                "list_of_records": [{"first_name": "Herbert", "age": 70}],
            },
            {
                "nested_record": {"nested_int": 5, "nested_float_list": [-2.0, 7.0]},
                "list_of_records": [
                    {"first_name": "Doug", "age": 55},
                    {"first_name": "Jess", "age": 66},
                    {"first_name": "Julia", "age": 30},
                ],
            },
            {
                "nested_record": {"nested_int": 7, "nested_float_list": [3.0, 4.0]},
                "list_of_records": [{"first_name": "Karl", "age": 32}],
            },
        ]
        features = {
            "nested_record.nested_int": tf.io.FixedLenFeature([], tf.dtypes.int32),
            "nested_record.nested_float_list[*]": tf.io.FixedLenFeature(
                [2], tf.dtypes.float32
            ),
            "list_of_records[0].first_name": tf.io.FixedLenFeature(
                [], tf.dtypes.string
            ),
        }
        expected_data = [
            {
                "nested_record.nested_int": tf.convert_to_tensor([0, 5, 7]),
                "nested_record.nested_float_list[*]": tf.convert_to_tensor(
                    [[0.0, 10.0], [-2.0, 7.0], [3.0, 4.0]]
                ),
                "list_of_records[0].first_name": tf.convert_to_tensor(
                    [
                        tf.compat.as_bytes("Herbert"),
                        tf.compat.as_bytes("Doug"),
                        tf.compat.as_bytes("Karl"),
                    ]
                ),
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=3,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_parse_map_entry(self):
        """test_parse_map_entry"""
        reader_schema = """
        {
           "type": "record",
           "name": "nesting",
           "fields": [
              {
                 "name": "map_of_records",
                 "type": {
                    "type": "map",
                    "values": {
                       "type": "record",
                       "name": "secondPerson",
                       "fields": [
                          {
                             "name": "first_name",
                             "type": "string"
                          },
                          {
                             "name": "age",
                             "type": "int"
                          }
                       ]
                    }
                 }
              }
           ]
        }
        """
        record_data = [
            {
                "map_of_records": {
                    "first": {"first_name": "Herbert", "age": 70},
                    "second": {"first_name": "Julia", "age": 30},
                }
            },
            {
                "map_of_records": {
                    "first": {"first_name": "Doug", "age": 55},
                    "second": {"first_name": "Jess", "age": 66},
                }
            },
            {
                "map_of_records": {
                    "first": {"first_name": "Karl", "age": 32},
                    "second": {"first_name": "Joan", "age": 21},
                }
            },
        ]
        # TODO(fraudies): Using FixedLenFeature([1], tf.dtypes.int32) this segfaults
        features = {
            "map_of_records['second'].age": tf.io.FixedLenFeature([], tf.dtypes.int32)
        }
        expected_data = [
            {"map_of_records['second'].age": tf.convert_to_tensor([30, 66, 21])}
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=3,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_parse_int_as_long_fail(self):
        """test_parse_int_as_long_fail"""
        schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "index",
                   "type": "int"
                }
             ]
          }
          """
        record_data = [{"index": 0}]
        features = {"index": tf.io.FixedLenFeature([], tf.dtypes.int64)}
        self._test_fail_dataset(schema, record_data, features, 1)

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_parse_int_as_sparse_type_fail(self):
        """test_parse_int_as_sparse_type_fail"""
        schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "index",
                   "type": "int"
                }
             ]
          }
          """
        record_data = [{"index": 5}]
        features = {
            "index": tf.io.SparseFeature(
                index_key="index", value_key="value", dtype=tf.dtypes.float32, size=10
            )
        }
        self._test_fail_dataset(schema, record_data, features, 1)

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_parse_float_as_double_fail(self):
        """test_parse_float_as_double_fail"""
        schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "weight",
                   "type": "float"
                }
             ]
          }
          """
        record_data = [{"weight": 0.5}]
        features = {"weight": tf.io.FixedLenFeature([], tf.dtypes.float64)}
        self._test_fail_dataset(schema, record_data, features, 1)

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_fixed_length_without_proper_default_fail(self):
        """test_fixed_length_without_proper_default_fail"""
        schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "int_list_type",
                   "type": {
                      "type":"array",
                      "items":"int"
                   }
                }
             ]
          }
          """
        record_data = [{"int_list_type": [0, 1, 2]}, {"int_list_type": [0, 1]}]
        features = {"int_list_type": tf.io.FixedLenFeature([], tf.dtypes.int32)}
        self._test_fail_dataset(schema, record_data, features, 1)

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_wrong_spelling_of_feature_name_fail(self):
        """test_wrong_spelling_of_feature_name_fail"""
        schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
               {"name": "int_type", "type": "int"}
             ]
          }"""
        record_data = [{"int_type": 0}]
        features = {"wrong_spelling": tf.io.FixedLenFeature([], tf.dtypes.int32)}
        self._test_fail_dataset(schema, record_data, features, 1)

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_wrong_index(self):
        """test_wrong_index"""
        schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "list_of_records",
                   "type": {
                      "type": "array",
                      "items": {
                         "type": "record",
                         "name": "person",
                         "fields": [
                            {
                               "name": "first_name",
                               "type": "string"
                            }
                         ]
                      }
                   }
                }
             ]
          }
          """
        record_data = [{"list_of_records": [{"first_name": "My name"}]}]
        features = {
            "list_of_records[2].name": tf.io.FixedLenFeature([], tf.dtypes.string)
        }
        self._test_fail_dataset(schema, record_data, features, 1)

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_filter_with_variable_length(self):
        """test_filter_with_variable_length"""
        reader_schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "guests",
                   "type": {
                      "type": "array",
                      "items": {
                         "type": "record",
                         "name": "person",
                         "fields": [
                            {
                               "name": "name",
                               "type": "string"
                            },
                            {
                               "name": "gender",
                               "type": "string"
                            }
                         ]
                      }
                   }
                }
             ]
          }
          """
        record_data = [
            {
                "guests": [
                    {"name": "Hans", "gender": "male"},
                    {"name": "Mary", "gender": "female"},
                    {"name": "July", "gender": "female"},
                ]
            },
            {
                "guests": [
                    {"name": "Joel", "gender": "male"},
                    {"name": "JoAn", "gender": "female"},
                    {"name": "Marc", "gender": "male"},
                ]
            },
        ]
        features = {
            "guests[gender='male'].name": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.string
            ),
            "guests[gender='female'].name": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.string
            ),
        }
        expected_data = [
            {
                "guests[gender='male'].name": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [1, 0], [1, 1]],
                    values=[
                        tf.compat.as_bytes("Hans"),
                        tf.compat.as_bytes("Joel"),
                        tf.compat.as_bytes("Marc"),
                    ],
                    dense_shape=[2, 2],
                ),
                "guests[gender='female'].name": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [0, 1], [1, 0]],
                    values=[
                        tf.compat.as_bytes("Mary"),
                        tf.compat.as_bytes("July"),
                        tf.compat.as_bytes("JoAn"),
                    ],
                    dense_shape=[2, 2],
                ),
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_filter_with_empty_result(self):
        """test_filter_with_empty_result"""
        reader_schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "guests",
                   "type": {
                      "type": "array",
                      "items": {
                         "type": "record",
                         "name": "person",
                         "fields": [
                            {
                               "name":"name",
                               "type":"string"
                            },
                            {
                               "name":"gender",
                               "type":"string"
                            }
                         ]
                      }
                   }
                }
             ]
          }
          """
        record_data = [
            {"guests": [{"name": "Hans", "gender": "male"}]},
            {"guests": [{"name": "Joel", "gender": "male"}]},
        ]
        features = {
            "guests[gender='wrong_value'].name": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.string
            )
        }
        expected_data = [
            {
                "guests[gender='wrong_value'].name": tf.compat.v1.SparseTensorValue(
                    indices=np.empty(shape=[0, 2], dtype=np.int64),
                    values=np.empty(shape=[0], dtype=np.str),
                    dense_shape=np.asarray([2, 0]),
                )
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_filter_with_wrong_key_fail(self):
        """test_filter_with_wrong_key_fail"""
        reader_schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "guests",
                   "type": {
                      "type": "array",
                      "items": {
                         "type": "record",
                         "name": "person",
                         "fields": [
                            {
                               "name":"name",
                               "type":"string"
                            }
                         ]
                      }
                   }
                }
             ]
          }
          """
        record_data = [{"guests": [{"name": "Hans"}]}]
        features = {
            "guests[wrong_key='female'].name": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.string
            )
        }
        self._test_fail_dataset(reader_schema, record_data, features, 1)

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_filter_with_wrong_pair_fail(self):
        """test_filter_with_wrong_pair_fail"""
        reader_schema = """
          {
             "type":"record",
             "name":"data_row",
             "fields":[
                {
                   "name":"guests",
                   "type":{
                      "type":"array",
                      "items":{
                         "type":"record",
                         "name":"person",
                         "fields":[
                            {
                               "name":"name",
                               "type":"string"
                            }
                         ]
                      }
                   }
                }
             ]
          }
          """
        record_data = [{"guests": [{"name": "Hans"}]}]
        features = {
            "guests[forgot_the_separator].name": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.string
            )
        }
        self._test_fail_dataset(reader_schema, record_data, features, 1)

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_filter_with_too_many_separators_fail(self):
        """test_filter_with_too_many_separators_fail"""
        reader_schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "guests",
                   "type": {
                      "type": "array",
                      "items": {
                         "type":"record",
                         "name":"person",
                         "fields":[
                            {
                               "name":"name",
                               "type":"string"
                            }
                         ]
                      }
                   }
                }
             ]
          }
          """
        record_data = [{"guests": [{"name": "Hans"}]}]
        features = {
            "guests[used=too=many=separators].name": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.string
            )
        }
        self._test_fail_dataset(reader_schema, record_data, features, 1)

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_filter_for_nested_record(self):
        """test_filter_for_nested_record"""
        reader_schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "guests",
                   "type": {
                      "type": "array",
                      "items": {
                         "type": "record",
                         "name": "person",
                         "fields": [
                            {
                               "name": "name",
                               "type": "string"
                            },
                            {
                               "name": "gender",
                               "type": "string"
                            },
                            {
                               "name": "address",
                               "type": {
                                  "type": "record",
                                  "name": "postal",
                                  "fields": [
                                     {
                                        "name":"street",
                                        "type":"string"
                                     },
                                     {
                                        "name":"zip",
                                        "type":"int"
                                     },
                                     {
                                        "name":"state",
                                        "type":"string"
                                     }
                                  ]
                               }
                            }
                         ]
                      }
                   }
                }
             ]
          }
          """
        record_data = [
            {
                "guests": [
                    {
                        "name": "Hans",
                        "gender": "male",
                        "address": {
                            "street": "California St",
                            "zip": 94040,
                            "state": "CA",
                        },
                    },
                    {
                        "name": "Mary",
                        "gender": "female",
                        "address": {"street": "Ellis St", "zip": 29040, "state": "MA"},
                    },
                ]
            }
        ]
        features = {
            "guests[gender='female'].address.street": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.string
            )
        }
        expected_data = [
            {
                "guests[gender='female']"
                + ".address.street": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0]],
                    values=[tf.compat.as_bytes("Ellis St")],
                    dense_shape=[1, 1],
                )
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_filter_with_bytes_as_type(self):
        """test_filter_with_bytes_as_type"""
        reader_schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
                {
                   "name": "guests",
                   "type": {
                      "type": "array",
                      "items": {
                         "type": "record",
                         "name": "person",
                         "fields": [
                            {
                               "name":"name",
                               "type":"bytes"
                            },
                            {
                               "name":"gender",
                               "type":"bytes"
                            }
                         ]
                      }
                   }
                }
             ]
          }
          """
        record_data = [
            {
                "guests": [
                    {"name": b"Hans", "gender": b"male"},
                    {"name": b"Mary", "gender": b"female"},
                    {"name": b"July", "gender": b"female"},
                ]
            },
            {
                "guests": [
                    {"name": b"Joel", "gender": b"male"},
                    {"name": b"JoAn", "gender": b"female"},
                    {"name": b"Marc", "gender": b"male"},
                ]
            },
        ]
        features = {
            "guests[gender='male'].name": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.string
            ),
            "guests[gender='female'].name": tfio.experimental.columnar.VarLenFeatureWithRank(
                tf.dtypes.string
            ),
        }
        expected_data = [
            {
                "guests[gender='male'].name": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [1, 0], [1, 1]],
                    values=[
                        tf.compat.as_bytes("Hans"),
                        tf.compat.as_bytes("Joel"),
                        tf.compat.as_bytes("Marc"),
                    ],
                    dense_shape=[2, 2],
                ),
                "guests[gender='female'].name": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [0, 1], [1, 0]],
                    values=[
                        tf.compat.as_bytes("Mary"),
                        tf.compat.as_bytes("July"),
                        tf.compat.as_bytes("JoAn"),
                    ],
                    dense_shape=[2, 2],
                ),
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )

    # @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_ignore_namespace(self):
        """test_namespace"""
        reader_schema = """
          {
            "namespace": "com.test",
            "type": "record",
            "name": "simple",
            "fields": [
                {
                   "name":"string_value",
                   "type":"string"
                }
            ]
          }"""
        features = {"string_value": tf.io.FixedLenFeature([], tf.dtypes.string)}
        record_data = [{"string_value": "a"}, {"string_value": "bb"}]
        expected_data = [
            {
                "string_value": tf.convert_to_tensor(
                    [tf.compat.as_bytes("a"), tf.compat.as_bytes("bb")]
                )
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_broken_schema_fail(self):
        """test_broken_schema_fail"""
        valid_schema = """
          {
            "type": "record",
            "name": "row",
            "fields": [
                {"name": "int_value", "type": "int"}
            ]
          }"""
        record_data = [{"int_value": 0}]
        broken_schema = """
          {
            "type": "record",
            "name": "row",
            "fields": [
                {"name": "index", "type": "int"},
                {"name": "boolean_type"}
            ]
          }"""
        features = {"index": tf.io.FixedLenFeature([], tf.dtypes.int64)}
        self._test_fail_dataset(
            valid_schema, record_data, features, 1, parser_schema=broken_schema
        )

    @pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
    def test_some_optimization_broke_string_repeats_in_batch(self):
        """test_some_optimization_broke_string_repeats_in_batch"""
        # In the past this test failed but now passes
        reader_schema = """
            {
              "type": "record",
              "name": "simple",
              "fields": [
                  {
                     "name":"string_value",
                     "type":"string"
                  }
              ]
            }"""
        features = {"string_value": tf.io.FixedLenFeature([], tf.dtypes.string)}
        record_data = [{"string_value": "aa"}, {"string_value": "bb"}]
        expected_data = [
            {
                "string_value": np.asarray(
                    [tf.compat.as_bytes("aa"), tf.compat.as_bytes("bb")]
                )
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )

    @pytest.mark.skip(reason="failed with tf 2.2 rc3 on linux")
    # Note current filters resolve to single item and we remove the dimension introduced by that
    def test_filter_of_sparse_feature(self):
        """test_filter_of_sparse_feature"""
        reader_schema = """
            {
               "type": "record",
               "name": "data_row",
               "fields": [
                  {
                     "name": "guests",
                     "type": {
                        "type": "array",
                        "items": {
                           "type": "record",
                           "name": "person",
                           "fields": [
                              {
                                 "name": "name",
                                 "type": "string"
                              },
                              {
                                 "name": "gender",
                                 "type": "string"
                              },
                              {
                                 "name": "address",
                                 "type": {
                                    "type": "array",
                                    "items": {
                                       "type": "record",
                                       "name": "postal",
                                       "fields": [
                                          {
                                             "name":"street",
                                             "type":"string"
                                          },
                                          {
                                             "name":"zip",
                                             "type":"long"
                                          },
                                          {
                                             "name":"street_no",
                                             "type":"int"
                                          }
                                       ]
                                    }
                                 }
                              }
                           ]
                        }
                     }
                  }
               ]
            }
            """
        record_data = [
            {
                "guests": [
                    {
                        "name": "Hans",
                        "gender": "male",
                        "address": [
                            {
                                "street": "California St",
                                "zip": 94040,
                                "state": "CA",
                                "street_no": 1,
                            },
                            {
                                "street": "New York St",
                                "zip": 32012,
                                "state": "NY",
                                "street_no": 2,
                            },
                        ],
                    },
                    {
                        "name": "Mary",
                        "gender": "female",
                        "address": [
                            {
                                "street": "Ellis St",
                                "zip": 29040,
                                "state": "MA",
                                "street_no": 3,
                            }
                        ],
                    },
                ]
            }
        ]
        features = {
            "guests[gender='female'].address": tf.io.SparseFeature(
                index_key="zip",
                value_key="street_no",
                dtype=tf.dtypes.int32,
                size=94040,
            )
        }
        # Note, the filter introduces an additional index,
        # because filters can have multiple items
        expected_data = [
            {
                "guests[gender='female'].address": tf.compat.v1.SparseTensorValue(
                    np.asarray([[0, 0, 29040]]),
                    np.asarray([3]),
                    np.asarray([1, 1, 94040]),
                )
            }
        ]
        self._test_pass_dataset(
            reader_schema=reader_schema,
            record_data=record_data,
            expected_data=expected_data,
            features=features,
            batch_size=2,
        )


if __name__ == "__main__":
    test.main()
