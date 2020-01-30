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

import numpy as np
from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.framework import ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.util import compat
from tensorflow_io.core.python.experimental.parse_avro_ops import parse_avro

from tensorflow_io.avro.python.utils.avro_serialization import AvroSerializer

from tensorflow_io.avro.python.tests.avro_dataset_test_base import AvroDatasetTestBase


class AvroDatasetTest(AvroDatasetTestBase):

    def _test_pass_dataset(self, reader_schema, record_data, expected_data, features):
        serializer = AvroSerializer(reader_schema)
        for expected_datum, record in zip(expected_data, record_data):
            actual = parse_avro(serialized=serializer.serialize(record),
                                reader_schema=reader_schema,
                                features=features)
            self.assertValuesEqual(expected=expected_datum,
                                   actual=actual)

    def test_primitive_types(self):
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
                "string_value": "SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?",
                "bytes_value": b"SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?",
                "double_value": -1.0,
                "float_value": -1.0,
                "long_value": 9223372036854775807,
                "int_value": 2147483648 - 1,
                "boolean_value": True,
            },
            {
                "string_value": "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
                "bytes_value": b"ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
                "double_value": 1.0,
                "float_value": 1.0,
                "long_value": -9223372036854775807 - 1,
                "int_value": -2147483648,
                "boolean_value": False,
            }
        ]
        features = {
            "string_value": parsing_ops.FixedLenFeature([], tf_types.string),
            "bytes_value": parsing_ops.FixedLenFeature([], tf_types.string),
            "double_value": parsing_ops.FixedLenFeature([], tf_types.float64),
            "float_value": parsing_ops.FixedLenFeature([], tf_types.float32),
            "long_value": parsing_ops.FixedLenFeature([], tf_types.int64),
            "int_value": parsing_ops.FixedLenFeature([], tf_types.int32),
            "boolean_value": parsing_ops.FixedLenFeature([], tf_types.bool)
        }
        expected_data = [
            {
                "string_value":
                    ops.convert_to_tensor(
                        np.asarray([
                            compat.as_bytes(""),
                            compat.as_bytes("SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"),
                            compat.as_bytes("ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789")
                        ])
                    ),
                "bytes_value":
                    ops.convert_to_tensor([
                        compat.as_bytes(""),
                        compat.as_bytes("SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"),
                        compat.as_bytes("ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789")
                    ]),
                # Note, conversion utils `ops.EagerTensor` only seems to support single precision.
                # Proper values for double precision are 1.7976931348623157e+308, -1.7976931348623157e+308
                # In addition, precision is not maintained by the conversion, thus, I simplify set 1.0
                # and -1.0 instead of proper values 3.40282306074e+38 and -3.40282306074e+38.
                "double_value":
                    ops.convert_to_tensor([
                        0.0,
                        -1.0,
                        1.0
                    ]),
                "float_value":
                    ops.convert_to_tensor([
                        0.0,
                        -1.0,
                        1.0
                    ]),
                "long_value":
                    ops.convert_to_tensor([
                        0,
                        9223372036854775807,
                        -9223372036854775807 - 1
                    ]),
                "int_value":
                    ops.convert_to_tensor([
                        0,
                        2147483648 - 1,
                        -2147483648
                    ]),
                "boolean_value":
                    ops.convert_to_tensor([
                        False,
                        True,
                        False
                    ])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features)
