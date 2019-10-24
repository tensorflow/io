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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
from tensorflow.python.platform import test
from tensorflow.python.framework import dtypes as tf_types
from tensorflow.python.ops import parsing_ops
from tensorflow.python.util import compat
from tensorflow.python.framework import sparse_tensor

from tensorflow_io.avro.python.tests import avro_dataset_test_base as \
    avro_test_base

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '50'


class AvroDatasetTest(avro_test_base.AvroDatasetTestBase):

    def test_primitive_types(self):
        writer_schema = """{
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
                "double_value": 1.7976931348623157e+308,
                "float_value": 3.40282306074e+38,
                "long_value": 9223372036854775807,
                "int_value": 2147483648 - 1,
                "boolean_value": True,
            },
            {
                "string_value": "ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
                "bytes_value": b"ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789",
                "double_value": -1.7976931348623157e+308,
                "float_value": -3.40282306074e+38,
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
                    np.asarray([
                        compat.as_bytes(""),
                        compat.as_bytes("SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"),
                        compat.as_bytes("ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789")
                    ]),
                "bytes_value":
                    np.asarray([
                        compat.as_bytes(""),
                        compat.as_bytes("SpecialChars@!#$%^&*()-_=+{}[]|/`~\\\'?"),
                        compat.as_bytes("ABCDEFGHIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwz0123456789")
                    ]),
                "double_value":
                    np.asarray([
                        0.0,
                        1.7976931348623157e+308,
                        -1.7976931348623157e+308
                    ]),
                "float_value":
                    np.asarray([
                        0.0,
                        3.40282306074e+38,
                        -3.40282306074e+38
                    ]),
                "long_value":
                    np.asarray([
                        0,
                        9223372036854775807,
                        -9223372036854775807 - 1
                    ]),
                "int_value":
                    np.asarray([
                        0,
                        2147483648 - 1,
                        -2147483648
                    ]),
                "boolean_value":
                    np.asarray([
                        False,
                        True,
                        False
                    ])
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=1)

    def test_fixed_enum_types(self):
        writer_schema = """{
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
                    np.asarray([
                        compat.as_bytes("0123456789"),
                        compat.as_bytes("1234567890"),
                        compat.as_bytes("2345678901")
                    ]),
                "enum_value":
                    np.asarray([
                        b"BLUE",
                        b"GREEN",
                        b"BROWN"
                    ])
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=1)

    def test_batching(self):
        writer_schema = """{
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
            {"int_value": np.asarray([0, 1])},
            {"int_value": np.asarray([2])}
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=2, num_epochs=1)

    def test_padding_from_default_with_drop_remainder(self):
        writer_schema = """{
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
        # Note, last batch is dropped
        expected_data = [
            {"fixed_len[*]": np.asarray([[0, 1], [1, 1], [2, 1]])}
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=None,
                                skip_out_of_range_error=True)

    def test_padding_from_default_without_drop_remainder(self):
        writer_schema = """{
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
            {"fixed_len[*]": np.asarray([[0, 1], [1, 1], [2, 1]])},
            {"fixed_len[*]": np.asarray([[3, 1]])}
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=1,
                                skip_out_of_range_error=True)

    def test_none_batch(self):
        writer_schema = """{
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
            "fixed_len[*]": parsing_ops.FixedLenFeature([], tf_types.int32,
                                                        default_value=0)
        }
        # Note, last batch is dropped
        expected_data = [
            {"fixed_len[*]": np.asarray([0, 1, 2])}
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=None,
                                skip_out_of_range_error=True)

    def test_batching_with_default(self):
        writer_schema = """{
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
            {"fixed_len[*]": np.asarray([
                [0, 1, 2],
                [3, 4, 5]])
            },
            {"fixed_len[*]": np.asarray([
                [6, 7, 8]])
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=2, num_epochs=1)

    def test_labels(self):
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "feature1", "type": "int"},
                  {"name": "label1", "type": "int"},
                  {"name": "label2", "type": "int"}
              ]}"""
        record_data = [
            {"feature1": 10, "label1": 0, "label2": 5},
            {"feature1": 20, "label1": 1, "label2": 6},
            {"feature1": 30, "label1": 2, "label2": 7}
        ]
        features = {
            "feature1": parsing_ops.FixedLenFeature([], tf_types.int32),
            "label1": parsing_ops.FixedLenFeature([], tf_types.int32),
            "label2": parsing_ops.FixedLenFeature([], tf_types.int32)
        }
        expected_data = [
            (
                {"feature1": np.asarray([10, 20, 30])},
                {"label1": np.asarray([0, 1, 2]), "label2": np.asarray([5, 6, 7])}
            )
        ]
        label_keys = ["label1", "label2"]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3,
                                num_epochs=1,
                                label_keys=label_keys)

    def test_larger_file_size(self):
        # Added because of existing implementations had bug if the file
        # size is larger than the buffer size
        n_data = 1024
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_value",
                     "type": "int"
                  }
              ]}"""
        record_data = [dict([("int_value", datum)]) for datum in range(n_data)]
        features = {
            "int_value": parsing_ops.FixedLenFeature([], tf_types.int32)
        }
        # Add batch dimension
        expected_data = [
            dict([("int_value", np.asarray([datum]))]) for datum in range(n_data)]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=1, num_epochs=1)

    def test_schema_projection(self):
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"},
                  {"name": "bool_value", "type": "boolean"}
              ]}"""
        reader_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"}
              ]}"""
        record_data = [
            {"int_value": 0, "bool_value": True},
            {"int_value": 1, "bool_value": False}
        ]
        features = {
            "int_value": parsing_ops.FixedLenFeature([], tf_types.int32)
        }
        expected_data = [
            {"int_value": np.asarray([0, 1])}
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=reader_schema,
                                batch_size=2, num_epochs=1)

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
        features = {
            "int_value": parsing_ops.FixedLenFeature([], tf_types.int64),
            "long_value": parsing_ops.FixedLenFeature([], tf_types.double)
        }
        expected_data = [
            {
                "int_value": np.asarray([0, 1]),
                "long_value": np.asarray([111.0, 222.0])
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=reader_schema,
                                batch_size=2, num_epochs=1)

    def test_null_union_primitive_type(self):
        writer_schema = """
      {
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
                    np.asarray([]),
                "multi_type:int":
                    np.asarray([]),
                "multi_type:long":
                    np.asarray([]),
                "multi_type:float":
                    np.asarray([]),
                "multi_type:double":
                    np.asarray([1.0, 2.0, 3.0, 1.0]),
                "multi_type:string":
                    np.asarray([compat.as_bytes("abc")])
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=6, num_epochs=1)

    def test_union_with_null(self):
        writer_schema = """
      {
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
                "possible_float_type:float": np.asarray([1.0, -1.0])
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=1)

    def test_fixed_length_list(self):
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
            {"int_list": [6, 7, 8]}
        ]
        features = {
            "int_list[*]": parsing_ops.FixedLenFeature([3], tf_types.int32)
        }
        expected_data = [
            {"int_list[*]": np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]])}
        ]

        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=1)

    def test_fixed_length_with_default_vector(self):
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
            {"int_list": [3]},
            {"int_list": [6, 7]}
        ]
        features = {
            "int_list[*]": parsing_ops.FixedLenFeature([3], tf_types.int32,
                                                       default_value=[0, 1, 2])
        }
        expected_data = [
            {"int_list[*]": np.asarray([
                [0, 1, 2],
                [3, 1, 2],
                [6, 7, 2]])
            }
        ]

        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=1)

    def test_fixed_length_with_default_scalar(self):
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
            {"int_list": [3]},
            {"int_list": [6, 7]}
        ]
        features = {
            "int_list[*]": parsing_ops.FixedLenFeature([], tf_types.int32,
                                                       default_value=0)
        }
        expected_data = [
            {"int_list[*]": np.asarray([
                [0, 1, 2],
                [3, 0, 0],
                [6, 7, 0]])
            }
        ]

        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=1)

    def test_dense_2d(self):
        writer_schema = """{
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
                 np.asarray([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])}
        ]

        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=2, num_epochs=1)

    def test_dense_array_3d(self):
        # Here we use arrays directly for the nesting
        writer_schema = """{
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
        ]
        features = {
            "int_list[*][*]": parsing_ops.FixedLenFeature([], tf_types.int32)
        }
        # Note, the outer dimension is the batch dimension
        expected_data = [
            {"int_list[*][*]": np.asarray([
                [[0, 1, 2], [10, 11, 12], [20, 21, 22]]
            ])},
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                batch_size=1, num_epochs=1,
                                reader_schema=writer_schema)

    def test_dense_record_3d(self):
        # Here we use records for the nesting
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_list",
                     "type": {
                        "type": "array",
                        "items":
                          {
                             "name" : "wrapper1",
                             "type" : "record",
                             "fields" : [
                                {
                                   "name": "nested_int_list",
                                   "type": {
                                      "type": "array",
                                      "items":
                                        {
                                           "name" : "wrapper2",
                                           "type" : "record",
                                           "fields" : [
                                              {
                                                 "name": "nested_nested_int_list",
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
                             ]
                          }
                     }
                  }
              ]}"""
        record_data = [
            {"int_list": [
                {"nested_int_list":
                    [
                        {"nested_nested_int_list": [1, 2, 3, 4]},
                        {"nested_nested_int_list": [5, 6, 7, 8]}
                    ]
                },
                {"nested_int_list":
                    [
                        {"nested_nested_int_list": [9, 10, 11, 12]},
                        {"nested_nested_int_list": [13, 14, 15, 16]}
                    ]
                },
                {"nested_int_list":
                    [
                        {"nested_nested_int_list": [17, 18, 19, 20]},
                        {"nested_nested_int_list": [21, 22, 23, 24]}
                    ]
                },
            ]}
        ]
        features = {
            "int_list[*].nested_int_list[*].nested_nested_int_list[*]":
                parsing_ops.FixedLenFeature([3, 2, 4], tf_types.int32)
        }
        expected_data = [
            {"int_list[*].nested_int_list[*].nested_nested_int_list[*]": np.asarray(
                [[
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8]
                    ],
                    [
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]
                    ],
                    [
                        [17, 18, 19, 20],
                        [21, 22, 23, 24]
                    ]
                ]]
            )},
        ]

        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=1, num_epochs=1)

    def test_sparse_feature(self):
        writer_schema = """{
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
                np.asarray([[0, 0], [0, 3], [1, 2]]),
                np.asarray([5.0, 2.0, 7.0]),
                np.asarray([2, 4]))},
            {"sparse_type": sparse_tensor.SparseTensorValue(
                np.asarray([[0, 1], [1, 3]]),
                np.asarray([6.0, 3.0]),
                np.asarray([2, 4]))}
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=2, num_epochs=1)

    def test_sparse_3d_feature(self):
        writer_schema = """{
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
                             "name":"first_index",
                             "type":"long"
                          },
                          {
                             "name":"second_index",
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
            {"sparse_type": [
                {"first_index": 0, "second_index": 1, "value": 5.0},
                {"first_index": 3, "second_index": 5, "value": 2.0}]
            },
            {"sparse_type": [
                {"first_index": 0, "second_index": 1, "value": 7.0}]
            }
        ]
        features = {
            "sparse_type": parsing_ops.SparseFeature(index_key=["first_index", "second_index"],
                                                     value_key="value",
                                                     dtype=tf_types.float32,
                                                     size=[4, 6])
        }
        expected_data = [
            {"sparse_type": sparse_tensor.SparseTensorValue(
                np.asarray([[0, 0, 1], [0, 3, 5], [1, 0, 1]]),
                np.asarray([5.0, 2.0, 7.0]),
                np.asarray([2, 4, 6]))}
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=2, num_epochs=1)

    def test_type_reuse(self):
        writer_schema = """
      {
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
                    np.asarray([[0, 0], [0, 3], [1, 0]]),
                    np.asarray([5.0, 2.0, 2.0]),
                    np.asarray([2, 4])),
                "second_value": sparse_tensor.SparseTensorValue(
                    np.asarray([[0, 2], [1, 1]]),
                    np.asarray([7.0, 2.0]),
                    np.asarray([2, 3]))
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=2, num_epochs=1)

    def test_variable_length(self):
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
                    np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0]]),
                    np.asarray([1, 2, 3, 4, 5, 6]),
                    np.asarray([3, 3])
                )
            }
        ]

        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=1)


    def test_variable_length_2d(self):
        writer_schema = """{
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
                    np.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0], [2, 0, 0]]),
                    np.asarray([1, 2, 3, 4, 5, 6, 6]),
                    np.asarray([3, 2, 3])
                )
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=1)

    def test_nesting(self):
        writer_schema = """
        {
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
                    np.asarray([0, 5, 7]),
                "nested_record.nested_float_list[*]":
                    np.asarray([[0.0, 10.0], [-2.0, 7.0], [3.0, 4.0]]),
                "list_of_records[0].first_name":
                    np.asarray([[compat.as_bytes("Herbert")],
                                [compat.as_bytes("Doug")],
                                [compat.as_bytes("Karl")]])
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=1)

    def test_parse_map_entry(self):
        writer_schema = """
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
                "map_of_records['second'].age": np.asarray([30, 66, 21])
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=3, num_epochs=1)

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
        self._test_fail_dataset(schema, record_data, features, schema, 1)

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
        self._test_fail_dataset(schema, record_data, features, schema, 1)

    def test_parse_float_as_double_fail(self):
        writer_schema = """
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
        self._test_fail_dataset(writer_schema, record_data, features,
                                writer_schema, 1)

    def test_fixed_length_without_proper_default_fail(self):
        writer_schema = """
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
        self._test_fail_dataset(writer_schema, record_data, features,
                                writer_schema, 1)

    def test_wrong_spelling_of_feature_name_fail(self):
        writer_schema = """
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
        self._test_fail_dataset(writer_schema, record_data, features,
                                writer_schema, 1)

    def test_wrong_index(self):
        writer_schema = """
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
        self._test_fail_dataset(writer_schema, record_data, features,
                                writer_schema, 1)

    def test_filter_with_variable_length(self):
        writer_schema = """
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
                        np.asarray([[0, 0], [1, 0], [1, 1]]),
                        np.asarray([compat.as_bytes("Hans"), compat.as_bytes("Joel"),
                                    compat.as_bytes("Marc")]),
                        np.asarray([2, 2])),
                "guests[gender='female'].name":
                    sparse_tensor.SparseTensorValue(
                        np.asarray([[0, 0], [0, 1], [1, 0]]),
                        np.asarray([compat.as_bytes("Mary"), compat.as_bytes("July"),
                                    compat.as_bytes("JoAn")]),
                        np.asarray([2, 2]))
            }
        ]

        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=2, num_epochs=1)

    def test_filter_with_empty_result(self):
        writer_schema = """
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
                        np.empty(shape=[0, 2], dtype=np.int64),
                        np.empty(shape=[0], dtype=np.str), np.asarray([2, 0]))
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=2, num_epochs=1)

    def test_filter_with_wrong_key_fail(self):
        writer_schema = """
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
        self._test_fail_dataset(writer_schema, record_data, features,
                                writer_schema, 1)

    def test_filter_with_wrong_pair_fail(self):
        writer_schema = """
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
        self._test_fail_dataset(writer_schema, record_data, features,
                                writer_schema, 1)

    def test_filter_with_too_many_separators_fail(self):
        writer_schema = """
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
        self._test_fail_dataset(writer_schema, record_data, features,
                                writer_schema, 1)

    def test_filter_for_nested_record(self):
        writer_schema = """
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
                        np.asarray([[0, 0]]), np.asarray([compat.as_bytes("Ellis St")]),
                        np.asarray([1, 1]))
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=2, num_epochs=1)

    def test_filter_with_bytes_as_type(self):
        writer_schema = """
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
                        np.asarray([[0, 0], [1, 0], [1, 1]]),
                        np.asarray([compat.as_bytes("Hans"), compat.as_bytes("Joel"),
                                    compat.as_bytes("Marc")]),
                        np.asarray([2, 2])),
                "guests[gender='female'].name":
                    sparse_tensor.SparseTensorValue(
                        np.asarray([[0, 0], [0, 1], [1, 0]]),
                        np.asarray([compat.as_bytes("Mary"), compat.as_bytes("July"),
                                    compat.as_bytes("JoAn")]),
                        np.asarray([2, 2]))
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=2, num_epochs=1)

    def test_namespace(self):
        writer_schema = """
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
                    np.asarray([
                        compat.as_bytes("a"),
                        compat.as_bytes("bb")
                    ])
            }
        ]
        self._test_pass_dataset(writer_schema=writer_schema,
                                record_data=record_data,
                                expected_data=expected_data,
                                features=features,
                                reader_schema=writer_schema,
                                batch_size=2, num_epochs=1)

    def test_broken_schema_fail(self):
        writer_schema = """
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
        self._test_fail_dataset(writer_schema, record_data, features,
                                reader_schema=broken_schema, batch_size=1)

    # TODO(fraudies): Fixme, returns aa, aa instead of aa, bb
    # def test_some_optimization_breaks_this(self):
    #   schema = """
    #     {
    #       "type": "record",
    #       "name": "simple",
    #       "fields": [
    #           {
    #              "name":"string_value",
    #              "type":"string"
    #           }
    #       ]
    #     }"""
    #   features = {
    #     "string_value": parsing_ops.FixedLenFeature([], tf_types.string)
    #   }
    #   record_data = [
    #     {
    #       "string_value": "aa"
    #     },
    #     {
    #       "string_value": "bb"
    #     }
    #   ]
    #   expected_data = [
    #     {
    #       "string_value":
    #         np.asarray([
    #           compat.as_bytes("aa"),
    #           compat.as_bytes("bb")
    #         ])
    #     }
    #   ]
    #   self._test_pass_dataset(writer_schema=schema,
    #                           record_data=record_data,
    #                           expected_data=expected_data,
    #                           features=features,
    #                           reader_schema=schema,
    #                           batch_size=2, num_epochs=1)

    def test_incompatible_schema_fail(self):
        writer_schema = """
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
        wrong_schema = """
      {
        "type": "record",
        "name": "row",
        "fields": [
            {"name": "index", "type": "int"},
            {"name": "crazy_type", "type": "boolean"}
        ]
      }"""
        features = {"index": parsing_ops.FixedLenFeature([], tf_types.int64)}
        self._test_fail_dataset(writer_schema, record_data, features,
                                reader_schema=wrong_schema, batch_size=1)

    # Not supported for now, will actually provide another dimension for filter
    # that can't be properly coerced
    # def test_filter_of_sparse_feature(self):
    #   writer_schema = """
    #     {
    #        "type": "record",
    #        "name": "data_row",
    #        "fields": [
    #           {
    #              "name": "guests",
    #              "type": {
    #                 "type": "array",
    #                 "items": {
    #                    "type": "record",
    #                    "name": "person",
    #                    "fields": [
    #                       {
    #                          "name": "name",
    #                          "type": "string"
    #                       },
    #                       {
    #                          "name": "gender",
    #                          "type": "string"
    #                       },
    #                       {
    #                          "name": "address",
    #                          "type": {
    #                             "type": "array",
    #                             "items": {
    #                                "type": "record",
    #                                "name": "postal",
    #                                "fields": [
    #                                   {
    #                                      "name":"street",
    #                                      "type":"string"
    #                                   },
    #                                   {
    #                                      "name":"zip",
    #                                      "type":"long"
    #                                   },
    #                                   {
    #                                      "name":"street_no",
    #                                      "type":"int"
    #                                   }
    #                                ]
    #                             }
    #                          }
    #                       }
    #                    ]
    #                 }
    #              }
    #           }
    #        ]
    #     }
    #     """
    #   record_data = [{
    #     "guests": [{
    #       "name":
    #         "Hans",
    #       "gender":
    #         "male",
    #       "address": [{
    #         "street": "California St",
    #         "zip": 94040,
    #         "state": "CA",
    #         "street_no": 1
    #       }, {
    #         "street": "New York St",
    #         "zip": 32012,
    #         "state": "NY",
    #         "street_no": 2
    #       }]
    #     }, {
    #       "name":
    #         "Mary",
    #       "gender":
    #         "female",
    #       "address": [{
    #         "street": "Ellis St",
    #         "zip": 29040,
    #         "state": "MA",
    #         "street_no": 3
    #       }]
    #     }]
    #   }]
    #   features = {
    #     "guests[gender='female'].address":
    #       parsing_ops.SparseFeature(
    #           index_key="zip",
    #           value_key="street_no",
    #           dtype=tf_types.int32,
    #           size=94040)
    #   }
    #   expected_data = [
    #     {
    #       "guests[gender='female'].address":
    #         sparse_tensor.SparseTensorValue(
    #             np.asarray([[0, 29040]]), np.asarray([3]),
    #             np.asarray([1, 94040]))
    #     }
    #   ]
    #   self._test_pass_dataset(writer_schema=writer_schema,
    #                           record_data=record_data,
    #                           expected_data=expected_data,
    #                           features=features,
    #                           batch_size=2, num_epochs=1)


if __name__ == "__main__":
    test.main()
