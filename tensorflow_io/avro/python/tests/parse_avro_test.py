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
from tensorflow.python.framework.errors import OpError
from tensorflow.python.ops import parsing_ops
from tensorflow.python.util import compat
from tensorflow.python.framework import sparse_tensor
from tensorflow_io.core.python.experimental.parse_avro_ops import parse_avro

from tensorflow_io.avro.python.utils.avro_serialization import AvroSerializer

from tensorflow_io.avro.python.tests.avro_dataset_test_base import AvroDatasetTestBase


class AvroDatasetTest(AvroDatasetTestBase):

    def assertDataEqual(self, expected, actual):
        for name, datum in expected.items():
            self.assertValuesEqual(expected=datum, actual=actual[name])

    @staticmethod
    def _batcher(iterable, step):
        n = len(iterable)
        for ndx in range(0, n, step):
            yield iterable[ndx:min(ndx + step, n)]

    def _test_pass_dataset(self, reader_schema, record_data, expected_data, features, batch_size):
        # Note, The batch size could be inferred from the expected data but found it better to be
        # explicit here
        serializer = AvroSerializer(reader_schema)
        for expected_datum, actual_records in zip(expected_data, AvroDatasetTest._batcher(record_data, batch_size)):
            # Get any key out of expected datum
            actual_datum = parse_avro(serialized=[serializer.serialize(r) for r in actual_records],
                                      reader_schema=reader_schema,
                                      features=features)
            self.assertDataEqual(expected=expected_datum,
                                 actual=actual_datum)

    def _test_fail_dataset(self, reader_schema, record_data, features, batch_size, **kwargs):
        parser_schema = kwargs.get("parser_schema", reader_schema)
        serializer = AvroSerializer(reader_schema)
        for actual_records in AvroDatasetTest._batcher(record_data, batch_size):
            # Get any key out of expected datum
            with self.assertRaises(OpError):
                _ = parse_avro(serialized=[serializer.serialize(r) for r in actual_records],
                               reader_schema=parser_schema,
                               features=features)

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
                                features=features,
                                batch_size=3)

    def test_fixed_enum_types(self):
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
            {
                "fixed_value": b"0123456789",
                "enum_value": "BLUE"
            },
            {
                "fixed_value": b"1234567890",
                "enum_value": "GREEN"
            },
            {
                "fixed_value": b"2345678901",
                "enum_value": "BROWN"
            }
        ]
        features = {
            "fixed_value": parsing_ops.FixedLenFeature([], tf_types.string),
            "enum_value": parsing_ops.FixedLenFeature([], tf_types.string)
        }
        expected_data = [
            {
                "fixed_value":
                    ops.convert_to_tensor([
                        compat.as_bytes("0123456789"),
                        compat.as_bytes("1234567890"),
                        compat.as_bytes("2345678901")
                    ]),
                "enum_value":
                    ops.convert_to_tensor([
                        b"BLUE",
                        b"GREEN",
                        b"BROWN"
                    ])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=3)

    def test_batching(self):
        reader_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"}
              ]}"""
        record_data = [
            {"int_value": 0},
            {"int_value": 1},
            {"int_value": 2}
        ]
        features = {
            "int_value": parsing_ops.FixedLenFeature([], tf_types.int32)
        }
        expected_data = [
            {"int_value": ops.convert_to_tensor([0, 1])},
            {"int_value": ops.convert_to_tensor([2])}
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)

    def test_padding_from_default(self):
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
            {"fixed_len": [3]}
        ]
        features = {
            "fixed_len[*]": parsing_ops.FixedLenFeature([2], tf_types.int32,
                                                        default_value=[0, 1])
        }
        # Note, last batch is NOT dropped
        expected_data = [
            {"fixed_len[*]": ops.convert_to_tensor([[0, 1], [1, 1], [2, 1]])},
            {"fixed_len[*]": ops.convert_to_tensor([[3, 1]])}
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=3)

    def test_batching_with_default(self):
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
            {"fixed_len": [6, 7, 8]}
        ]
        features = {
            "fixed_len[*]": parsing_ops.FixedLenFeature([3], tf_types.int32,
                                                        default_value=[0, 1, 2])
        }
        expected_data = [
            {"fixed_len[*]": ops.convert_to_tensor([
                [0, 1, 2],
                [3, 4, 5]])
            },
            {"fixed_len[*]": ops.convert_to_tensor([
                [6, 7, 8]])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)

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
            {
                "multi_type": 1.0
            },
            {
                "multi_type": 2.0
            },
            {
                "multi_type": 3.0
            },
            {
                "multi_type": True  # converted by py avro implementation into 1
            },
            {
                "multi_type": "abc"
            },
            {
                "multi_type": None
            }
        ]
        features = {
            "multi_type:boolean": parsing_ops.FixedLenFeature([], tf_types.bool),
            "multi_type:int": parsing_ops.FixedLenFeature([], tf_types.int32),
            "multi_type:long": parsing_ops.FixedLenFeature([], tf_types.int64),
            "multi_type:float": parsing_ops.FixedLenFeature([], tf_types.float32),
            "multi_type:double": parsing_ops.FixedLenFeature([], tf_types.float64),
            "multi_type:string": parsing_ops.FixedLenFeature([], tf_types.string)
        }
        expected_data = [
            {
                "multi_type:boolean":
                    ops.convert_to_tensor([]),
                "multi_type:int":
                    ops.convert_to_tensor([]),
                "multi_type:long":
                    ops.convert_to_tensor([]),
                "multi_type:float":
                    ops.convert_to_tensor([]),
                "multi_type:double":
                    ops.convert_to_tensor([1.0, 2.0, 3.0, 1.0]),
                "multi_type:string":
                    ops.convert_to_tensor([compat.as_bytes("abc")])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=6)

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
            {
                "possible_float_type": 1.0
            },
            {
                "possible_float_type": None
            },
            {
                "possible_float_type": -1.0
            }
        ]
        features = {
            "possible_float_type:float": parsing_ops.FixedLenFeature([],
                                                                     tf_types.float32)
        }
        # TODO(fraudies): If we have a default, then we use that in the place of
        #  the None
        expected_data = [
            {
                "possible_float_type:float": ops.convert_to_tensor([1.0, -1.0])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=3)

    def test_fixed_length_list(self):
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
            {"int_list": [6, 7, 8]}
        ]
        features = {
            "int_list[*]": parsing_ops.FixedLenFeature([3], tf_types.int32)
        }
        expected_data = [
            {"int_list[*]": ops.convert_to_tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])}
        ]

        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=3)

    def test_fixed_length_with_default_vector(self):
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
            {"int_list": [3]},
            {"int_list": [6, 7]}
        ]
        features = {
            "int_list[*]": parsing_ops.FixedLenFeature([3], tf_types.int32,
                                                       default_value=[0, 1, 2])
        }
        expected_data = [
            {"int_list[*]": ops.convert_to_tensor([
                [0, 1, 2],
                [3, 1, 2],
                [6, 7, 2]])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=3)

    def test_fixed_length_with_default_scalar(self):
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
            {"int_list": [3]},
            {"int_list": [6, 7]}
        ]
        features = {
            "int_list[*]": parsing_ops.FixedLenFeature([None], tf_types.int32,
                                                       default_value=0)
        }
        expected_data = [
            {"int_list[*]": ops.convert_to_tensor([
                [0, 1, 2],
                [3, 0, 0],
                [6, 7, 0]])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=3)

    def test_dense_2d(self):
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
            {"int_list": [
                {"nested_int_list": [1, 2, 3]},
                {"nested_int_list": [4, 5, 6]}
            ]},
            {"int_list": [
                {"nested_int_list": [7, 8, 9]},
                {"nested_int_list": [10, 11, 12]}
            ]}
        ]
        features = {
            "int_list[*].nested_int_list[*]":
                parsing_ops.FixedLenFeature([2, 3], tf_types.int32)
        }
        expected_data = [
            {"int_list[*].nested_int_list[*]":
                 ops.convert_to_tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])}
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)

    def test_dense_array_3d(self):
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
            "int_list[*][*]": parsing_ops.FixedLenFeature([None, None], tf_types.int32)
        }
        # Note, the outer dimension is the batch dimension
        expected_data = [
            {"int_list[*][*]": ops.convert_to_tensor([
                [[0, 1, 2], [10, 11, 12], [20, 21, 22]],
                [[1, 2, 3], [11, 12, 13], [21, 22, 23]]
            ])},
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)

    def test_sparse_feature(self):
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
            {"sparse_type": [{"index": 3, "value": 3.0}]}
        ]
        features = {
            "sparse_type": parsing_ops.SparseFeature(index_key="index",
                                                     value_key="value",
                                                     dtype=tf_types.float32,
                                                     size=4)
        }
        expected_data = [
            {"sparse_type": sparse_tensor.SparseTensorValue(
                indices=[[0, 0], [0, 3], [1, 2]],
                values=[5.0, 2.0, 7.0],
                dense_shape=[2, 4])},
            {"sparse_type": sparse_tensor.SparseTensorValue(
                indices=[[0, 1], [1, 3]],
                values=[6.0, 3.0],
                dense_shape=[2, 4])}
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)

    def test_type_reuse(self):
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
                "second_value": [{"index": 2, "value": 7.0}]
            },
            {
                "first_value": [{"index": 0, "value": 2.0}],
                "second_value": [{"index": 1, "value": 2.0}]
            }
        ]
        features = {
            "first_value": parsing_ops.SparseFeature(index_key="index",
                                                     value_key="value",
                                                     dtype=tf_types.float32,
                                                     size=4),
            "second_value": parsing_ops.SparseFeature(index_key="index",
                                                      value_key="value",
                                                      dtype=tf_types.float32,
                                                      size=3)
        }
        expected_data = [
            {
                "first_value": sparse_tensor.SparseTensorValue(
                    indices=[[0, 0], [0, 3], [1, 0]],
                    values=[5.0, 2.0, 2.0],
                    dense_shape=[2, 4]),
                "second_value": sparse_tensor.SparseTensorValue(
                    indices=[[0, 2], [1, 1]],
                    values=[7.0, 2.0],
                    dense_shape=[2, 3])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)

    def test_variable_length(self):
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
            {"int_list": [1, 2]},
            {"int_list": [3, 4, 5]},
            {"int_list": [6]}
        ]
        features = {
            'int_list[*]': parsing_ops.VarLenFeature(tf_types.int32)
        }
        expected_data = [
            {"int_list[*]":
                sparse_tensor.SparseTensorValue(
                    indices=[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0]],
                    values=[1, 2, 3, 4, 5, 6],
                    dense_shape=[3, 3]
                )
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=3)

    def test_variable_length_2d(self):
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
            'int_list_list[*][*]': parsing_ops.VarLenFeature(tf_types.int32)
        }
        expected_data = [
            {"int_list_list[*][*]":
                sparse_tensor.SparseTensorValue(
                    indices=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0], [2, 0, 0]],
                    values=[1, 2, 3, 4, 5, 6, 6],
                    dense_shape=[3, 2, 3]
                )
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=3)

    def test_nesting(self):
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
                "nested_record": {
                    "nested_int": 0,
                    "nested_float_list": [0.0, 10.0]
                },
                "list_of_records": [{
                    "first_name": "Herbert",
                    "age": 70
                }]
            },
            {
                "nested_record": {
                    "nested_int": 5,
                    "nested_float_list": [-2.0, 7.0]
                },
                "list_of_records": [{
                    "first_name": "Doug",
                    "age": 55
                }, {
                    "first_name": "Jess",
                    "age": 66
                }, {
                    "first_name": "Julia",
                    "age": 30
                }]
            },
            {
                "nested_record": {
                    "nested_int": 7,
                    "nested_float_list": [3.0, 4.0]
                },
                "list_of_records": [{
                    "first_name": "Karl",
                    "age": 32
                }]
            }
        ]
        features = {
            "nested_record.nested_int": parsing_ops.FixedLenFeature([], tf_types.int32),
            "nested_record.nested_float_list[*]": parsing_ops.FixedLenFeature([2], tf_types.float32),
            "list_of_records[0].first_name": parsing_ops.FixedLenFeature([1], tf_types.string)
        }
        expected_data = [
            {
                "nested_record.nested_int":
                    ops.convert_to_tensor([0, 5, 7]),
                "nested_record.nested_float_list[*]":
                    ops.convert_to_tensor([[0.0, 10.0], [-2.0, 7.0], [3.0, 4.0]]),
                "list_of_records[0].first_name":
                    ops.convert_to_tensor([compat.as_bytes("Herbert"),
                                           compat.as_bytes("Doug"),
                                           compat.as_bytes("Karl")])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=3)

    def test_parse_map_entry(self):
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
                    "first": {
                        "first_name": "Herbert",
                        "age": 70
                    },
                    "second": {
                        "first_name": "Julia",
                        "age": 30
                    }
                }
            },
            {
                "map_of_records": {
                    "first": {
                        "first_name": "Doug",
                        "age": 55
                    },
                    "second": {
                        "first_name": "Jess",
                        "age": 66
                    }
                }
            },
            {
                "map_of_records": {
                    "first": {
                        "first_name": "Karl",
                        "age": 32
                    },
                    "second": {
                        "first_name": "Joan",
                        "age": 21
                    }
                }
            }
        ]
        # TODO(fraudies): Using FixedLenFeature([1], tf_types.int32) this segfaults
        features = {
            "map_of_records['second'].age": parsing_ops.FixedLenFeature([], tf_types.int32)
        }
        expected_data = [
            {
                "map_of_records['second'].age": ops.convert_to_tensor([30, 66, 21])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=3)

    def test_parse_int_as_long_fail(self):
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
        features = {"index": parsing_ops.FixedLenFeature([], tf_types.int64)}
        self._test_fail_dataset(schema, record_data, features, 1)

    def test_parse_int_as_sparse_type_fail(self):
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
            "index":
                parsing_ops.SparseFeature(
                    index_key="index",
                    value_key="value",
                    dtype=tf_types.float32,
                    size=10)
        }
        self._test_fail_dataset(schema, record_data, features, 1)

    def test_parse_float_as_double_fail(self):
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
        features = {"weight": parsing_ops.FixedLenFeature([], tf_types.float64)}
        self._test_fail_dataset(schema, record_data, features, 1)

    def test_fixed_length_without_proper_default_fail(self):
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
        record_data = [
            {
                "int_list_type": [0, 1, 2]
            },
            {
                "int_list_type": [0, 1]
            }
        ]
        features = {
            "int_list_type": parsing_ops.FixedLenFeature([], tf_types.int32)
        }
        self._test_fail_dataset(schema, record_data, features, 1)

    def test_wrong_spelling_of_feature_name_fail(self):
        schema = """
          {
             "type": "record",
             "name": "data_row",
             "fields": [
               {"name": "int_type", "type": "int"}
             ]
          }"""
        record_data = [{"int_type": 0}]
        features = {
            "wrong_spelling": parsing_ops.FixedLenFeature([], tf_types.int32)
        }
        self._test_fail_dataset(schema, record_data, features, 1)

    def test_wrong_index(self):
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
        record_data = [{
            "list_of_records": [{
                "first_name": "My name"
            }]
        }]
        features = {
            "list_of_records[2].name":
                parsing_ops.FixedLenFeature([], tf_types.string)
        }
        self._test_fail_dataset(schema, record_data, features, 1)

    def test_filter_with_variable_length(self):
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
                    {
                        "name": "Hans",
                        "gender": "male"
                    },
                    {
                        "name": "Mary",
                        "gender": "female"
                    },
                    {
                        "name": "July",
                        "gender": "female"
                    }
                ]
            },
            {
                "guests": [
                    {
                        "name": "Joel",
                        "gender": "male"
                    }, {
                        "name": "JoAn",
                        "gender": "female"
                    }, {
                        "name": "Marc",
                        "gender": "male"
                    }
                ]
            }
        ]
        features = {
            "guests[gender='male'].name":
                parsing_ops.VarLenFeature(tf_types.string),
            "guests[gender='female'].name":
                parsing_ops.VarLenFeature(tf_types.string)
        }
        expected_data = [
            {
                "guests[gender='male'].name":
                    sparse_tensor.SparseTensorValue(
                        indices=[[0, 0], [1, 0], [1, 1]],
                        values=[compat.as_bytes("Hans"), compat.as_bytes("Joel"),
                                compat.as_bytes("Marc")],
                        dense_shape=[2, 2]),
                "guests[gender='female'].name":
                    sparse_tensor.SparseTensorValue(
                        indices=[[0, 0], [0, 1], [1, 0]],
                        values=[compat.as_bytes("Mary"), compat.as_bytes("July"),
                                compat.as_bytes("JoAn")],
                        dense_shape=[2, 2])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)

    def test_filter_with_empty_result(self):
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
        record_data = [{
            "guests": [{
                "name": "Hans",
                "gender": "male"
            }]
        }, {
            "guests": [{
                "name": "Joel",
                "gender": "male"
            }]
        }]
        features = {
            "guests[gender='wrong_value'].name":
                parsing_ops.VarLenFeature(tf_types.string)
        }
        expected_data = [
            {
                "guests[gender='wrong_value'].name":
                    sparse_tensor.SparseTensorValue(
                        indices=np.empty(shape=[0, 2], dtype=np.int64),
                        values=np.empty(shape=[0], dtype=np.str),
                        dense_shape=np.asarray([2, 0]))
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)

    def test_filter_with_wrong_key_fail(self):
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
        record_data = [{
            "guests": [{
                "name": "Hans"
            }]
        }]
        features = {
            "guests[wrong_key='female'].name":
                parsing_ops.VarLenFeature(tf_types.string)
        }
        self._test_fail_dataset(reader_schema, record_data, features, 1)

    def test_filter_with_wrong_pair_fail(self):
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
        record_data = [{
            "guests": [{
                "name": "Hans"
            }]
        }]
        features = {
            "guests[forgot_the_separator].name":
                parsing_ops.VarLenFeature(tf_types.string)
        }
        self._test_fail_dataset(reader_schema, record_data, features, 1)

    def test_filter_with_too_many_separators_fail(self):
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
        record_data = [{
            "guests": [{
                "name": "Hans"
            }]
        }]
        features = {
            "guests[used=too=many=separators].name":
                parsing_ops.VarLenFeature(tf_types.string)
        }
        self._test_fail_dataset(reader_schema, record_data, features, 1)

    def test_filter_for_nested_record(self):
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
        record_data = [{
            "guests": [{
                "name": "Hans",
                "gender": "male",
                "address": {
                    "street": "California St",
                    "zip": 94040,
                    "state": "CA"
                }
            }, {
                "name": "Mary",
                "gender": "female",
                "address": {
                    "street": "Ellis St",
                    "zip": 29040,
                    "state": "MA"
                }
            }]
        }]
        features = {
            "guests[gender='female'].address.street":
                parsing_ops.VarLenFeature(tf_types.string)
        }
        expected_data = [
            {
                "guests[gender='female'].address.street":
                    sparse_tensor.SparseTensorValue(
                        indices=[[0, 0]],
                        values=[compat.as_bytes("Ellis St")],
                        dense_shape=[1, 1])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)

    def test_filter_with_bytes_as_type(self):
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
        record_data = [{
            "guests": [{
                "name": b"Hans",
                "gender": b"male"
            }, {
                "name": b"Mary",
                "gender": b"female"
            }, {
                "name": b"July",
                "gender": b"female"
            }]
        }, {
            "guests": [{
                "name": b"Joel",
                "gender": b"male"
            }, {
                "name": b"JoAn",
                "gender": b"female"
            }, {
                "name": b"Marc",
                "gender": b"male"
            }]
        }]
        features = {
            "guests[gender='male'].name":
                parsing_ops.VarLenFeature(tf_types.string),
            "guests[gender='female'].name":
                parsing_ops.VarLenFeature(tf_types.string)
        }
        expected_data = [
            {
                "guests[gender='male'].name":
                    sparse_tensor.SparseTensorValue(
                        indices=[[0, 0], [1, 0], [1, 1]],
                        values=[compat.as_bytes("Hans"), compat.as_bytes("Joel"),
                                compat.as_bytes("Marc")],
                        dense_shape=[2, 2]),
                "guests[gender='female'].name":
                    sparse_tensor.SparseTensorValue(
                        indices=[[0, 0], [0, 1], [1, 0]],
                        values=[compat.as_bytes("Mary"), compat.as_bytes("July"),
                                compat.as_bytes("JoAn")],
                        dense_shape=[2, 2])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)

    def test_namespace(self):
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
        features = {
            "com.test.string_value": parsing_ops.FixedLenFeature([], tf_types.string)
        }
        record_data = [
            {
                "string_value": "a"
            },
            {
                "string_value": "bb"
            }
        ]
        expected_data = [
            {
                "com.test.string_value":
                    ops.convert_to_tensor([
                        compat.as_bytes("a"),
                        compat.as_bytes("bb")
                    ])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)

    def test_broken_schema_fail(self):
        valid_schema = """
          {
            "type": "record",
            "name": "row",
            "fields": [
                {"name": "int_value", "type": "int"}
            ]
          }"""
        record_data = [
            {"int_value": 0}
        ]
        broken_schema = """
          {
            "type": "record",
            "name": "row",
            "fields": [
                {"name": "index", "type": "int"},
                {"name": "boolean_type"}
            ]
          }"""
        features = {"index": parsing_ops.FixedLenFeature([], tf_types.int64)}
        self._test_fail_dataset(valid_schema, record_data, features, 1,
                                parser_schema=broken_schema)

    def test_some_optimization_broke_string_repeats_in_batch(self):
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
        features = {
            "string_value": parsing_ops.FixedLenFeature([], tf_types.string)
        }
        record_data = [
            {
                "string_value": "aa"
            },
            {
                "string_value": "bb"
            }
        ]
        expected_data = [
            {
                "string_value":
                    np.asarray([
                        compat.as_bytes("aa"),
                        compat.as_bytes("bb")
                    ])
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)

    # Note current filters resolve to single item and we remove the dimension introduced by that
    def test_filter_of_sparse_feature(self):
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
        record_data = [{
            "guests": [{
                "name":
                    "Hans",
                "gender":
                    "male",
                "address": [{
                    "street": "California St",
                    "zip": 94040,
                    "state": "CA",
                    "street_no": 1
                }, {
                    "street": "New York St",
                    "zip": 32012,
                    "state": "NY",
                    "street_no": 2
                }]
            }, {
                "name":
                    "Mary",
                "gender":
                    "female",
                "address": [{
                    "street": "Ellis St",
                    "zip": 29040,
                    "state": "MA",
                    "street_no": 3
                }]
            }]
        }]
        features = {
            "guests[gender='female'].address":
                parsing_ops.SparseFeature(
                    index_key="zip",
                    value_key="street_no",
                    dtype=tf_types.int32,
                    size=94040)
        }
        # Note, the filter introduces an additional index,
        # because filters can have multiple items
        expected_data = [
            {
                "guests[gender='female'].address":
                    sparse_tensor.SparseTensorValue(
                        np.asarray([[0, 0, 29040]]),
                        np.asarray([3]),
                        np.asarray([1, 1, 94040]))
            }
        ]
        self._test_pass_dataset(reader_schema=reader_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=2)
