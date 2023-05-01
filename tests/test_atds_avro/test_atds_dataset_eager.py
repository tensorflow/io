# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
import pytest
import tempfile
import re
import os
import gzip
import json
import numpy as np
import tensorflow as tf
import itertools
import snappy
import random
import avro.schema
from avro.datafile import DataFileWriter
from avro.io import DatumWriter
from parameterized import parameterized
from tensorflow.python.framework import errors
from tests.test_parse_avro_eager import AvroDatasetTestBase, AvroRecordsToFile
from tensorflow_io.python.ops import core_ops
from tensorflow_io.python.experimental.atds.dataset import ATDSDataset
from tensorflow_io.python.experimental.atds.features import (
    DenseFeature,
    SparseFeature,
    VarlenFeature,
)

"This file holds the test cases for ATDSDataset."


def create_atds_dataset(
    writer_schema,
    record_data,
    features,
    batch_size,
    drop_remainder=None,
    codec="deflate",
    num_parallel_calls=None,
):
    """
    Creates ATDSDataset by
    1. Generate Avro files with the writer_schema and record_data. Note: This uses DEFLATE codec.
    2. Create ATDSDataset with the generated files, batch size,
       and features config.
    """
    filename = os.path.join(tempfile.mkdtemp(), "test.avro")
    writer = AvroRecordsToFile(
        filename=filename, writer_schema=writer_schema, codec=codec
    )
    writer.write_records(record_data)
    return ATDSDataset(
        filenames=[filename],
        batch_size=batch_size,
        features=features,
        drop_remainder=drop_remainder,
        num_parallel_calls=num_parallel_calls,
    )


@pytest.mark.parametrize(
    ["record_data", "error_message"],
    [
        (
            [{"int_list_list": [[1, 2, 3], [3, 4]]}],
            "Failed to decode feature int_list_list. "
            "Reason: Number of decoded value 2 does not match the expected dimension size 3"
            " at the 2th dimension in user defined shape [2,3]",
        ),
        (
            [{"int_list_list": [[1, 2, 3], [3, 4, 5], [6, 7, 8], [9, 10, 11]]}],
            "Failed to decode feature int_list_list. "
            "Reason: Number of decoded value 4 does not match the expected dimension size 2"
            " at the 1th dimension in user defined shape [2,3]",
        ),
    ],
)
def test_dense_feature_decode_error(record_data, error_message):
    """test_dense_feature_decode_error"""
    schema = """{
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
    features = {"int_list_list": DenseFeature([2, 3], tf.dtypes.int32)}
    with pytest.raises(errors.InvalidArgumentError, match=re.escape(error_message)):
        dataset = create_atds_dataset(
            writer_schema=schema,
            record_data=record_data,
            features=features,
            batch_size=1,
        )
        iterator = iter(dataset)
        next(iterator)


@pytest.mark.parametrize(
    ["record_data", "error_message"],
    [
        (
            [{"int_list_list": [[1, 2, 3], [3, 4]]}],
            "Failed to decode feature int_list_list. "
            "Reason: Number of decoded value 2 does not match the expected dimension size 3"
            " at the 2th dimension in user defined shape [?,3]",
        ),
        (
            [{"int_list_list": [[]]}],
            "Failed to decode feature int_list_list. "
            "Reason: Number of decoded value 0 does not match the expected dimension size 3"
            " at the 2th dimension in user defined shape [?,3]",
        ),
    ],
)
def test_varlen_feature_decode_error(record_data, error_message):
    """test_varlen_feature_decode_error"""
    schema = """{
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
    features = {"int_list_list": VarlenFeature([-1, 3], tf.dtypes.int32)}
    with pytest.raises(errors.InvalidArgumentError, match=re.escape(error_message)):
        dataset = create_atds_dataset(
            writer_schema=schema,
            record_data=record_data,
            features=features,
            batch_size=1,
        )
        iterator = iter(dataset)
        next(iterator)


@pytest.mark.parametrize(
    ["record_data", "error_message"],
    [
        (
            [{"sparse_key": {"indices0": [0, 1], "values": []}}],
            "Failed to decode feature sparse_key. "
            "Reason: Numbers of decoded value in indice and values array are different. "
            "Numbers of decoded value in [indices0, values] arrays are [2, 0]",
        ),
        (
            [{"sparse_key": {"indices0": [0, 1, 2], "values": [0.5, -0.5]}}],
            "Failed to decode feature sparse_key. "
            "Reason: Numbers of decoded value in indice and values array are different. "
            "Numbers of decoded value in [indices0, values] arrays are [3, 2]",
        ),
    ],
)
def test_sparse_feature_decode_error(record_data, error_message):
    schema = """{
        "type": "record",
        "name": "row",
        "fields": [
            {
               "name": "sparse_key",
               "type" : {
                   "type" : "record",
                   "name" : "SparseTensor",
                   "fields" : [ {
                     "name" : "indices0",
                     "type" : {
                       "type" : "array",
                       "items" : "long"
                     }
                   }, {
                     "name" : "values",
                     "type" : {
                       "type" : "array",
                       "items" : "float"
                     }
                   } ]
               }
            }
        ]}"""
    features = {"sparse_key": SparseFeature(shape=[10], dtype=tf.dtypes.float32)}
    with pytest.raises(errors.InvalidArgumentError, match=re.escape(error_message)):
        dataset = create_atds_dataset(
            writer_schema=schema,
            record_data=record_data,
            features=features,
            batch_size=1,
        )
        iterator = iter(dataset)
        next(iterator)


@pytest.mark.parametrize(
    ["schema", "features", "record_data", "error_message"],
    [
        # test_dense_feature_non_nested_arrays
        (
            """{
                "type": "record",
                "name": "outer_record",
                "fields": [
                    {
                        "name": "non_nested_arrays",
                        "type": {
                            "type": "array",
                            "items": {
                                "type": "record",
                                "name": "inner_record",
                                "fields": [
                                    {
                                        "name": "inner_list",
                                        "type": {
                                            "type": "array",
                                            "items": "int"
                                        }
                                    }
                                ]
                            }
                        }
                    }
                ]}""",
            {"non_nested_arrays": DenseFeature([2, 2], tf.dtypes.int32)},
            [{"non_nested_arrays": [{"inner_list": [1, 2]}]}],
            "Dense feature 'non_nested_arrays' must be non-nullable nested arrays only. "
            "Invalid schema found:",
        ),
        # test_dense_feature_nullable_array
        (
            """{
                "type": "record",
                "name": "outer_record",
                "fields": [
                    {
                        "name": "nullable_array",
                        "type": {
                            "type": "array",
                            "items": ["null", {
                                    "type": "array",
                                    "items": "int"
                                }]
                        }
                    }
                ]}""",
            {"nullable_array": DenseFeature([2, 2], tf.dtypes.int32)},
            [{"nullable_array": [[1, 2], [3, 4]]}],
            "Dense feature 'nullable_array' must be non-nullable nested arrays only. "
            "Invalid schema found:",
        ),
        # test_dense_feature_type_mismatch
        (
            """{
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
                ]}""",
            {"int_list_list": DenseFeature([2, 2], tf.dtypes.int64)},
            [{"int_list_list": [[1, 2]]}],
            "Schema value type and metadata type mismatch in feature 'int_list_list'. "
            "Avro schema data type: int, metadata type: int64. "
            "Invalid schema found:",
        ),
        # test_dense_feature_rank_mismatch
        (
            """{
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
                ]}""",
            {"int_list_list": DenseFeature([1, 1, 2], tf.dtypes.int32)},
            [{"int_list_list": [[1, 2], [3, 4]]}],
            "Mismatch between avro schema rank and metadata rank in feature 'int_list_list'. "
            "Avro schema rank: 2, metadata rank: 3. "
            "Invalid schema found:",
        ),
        # test_varlen_feature_non_nested_arrays
        (
            """{
                "type": "record",
                "name": "outer_record",
                "fields": [
                    {
                        "name": "int_list_list",
                        "type": {
                            "type": "array",
                            "items": {
                                "type": "record",
                                "name": "inner_record",
                                "fields": [
                                    {
                                        "name": "inner_list",
                                        "type": {
                                            "type": "array",
                                            "items": "int"
                                        }
                                    }
                                ]
                            }
                        }
                    }
                ]}""",
            {"int_list_list": VarlenFeature([-1, 3], tf.dtypes.int32)},
            [{"int_list_list": [{"inner_list": [1, 2, 3]}, {"inner_list": [3, 4, 5]}]}],
            "Varlen feature 'int_list_list' must be non-nullable nested arrays only. "
            "Invalid schema found:",
        ),
        # test_varlen_feature_nullable_array
        (
            """{
                "type": "record",
                "name": "outer_record",
                "fields": [
                    {
                        "name": "nullable_array",
                        "type": {
                            "type": "array",
                            "items": ["null", {
                                "type": "array",
                                "items": "int"
                            }]
                        }
                    }
                ]}""",
            {"nullable_array": VarlenFeature([2, -1], tf.dtypes.int32)},
            [{"nullable_array": [[1, 2], [3]]}],
            "Varlen feature 'nullable_array' must be non-nullable nested arrays only. "
            "Invalid schema found:",
        ),
        # test_varlen_feature_type_mismatch
        (
            """{
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
                ]}""",
            {"int_list_list": VarlenFeature([2, -1], tf.dtypes.int64)},
            [{"int_list_list": [[1, 2], [1]]}],
            "Schema value type and metadata type mismatch in feature 'int_list_list'. "
            "Avro schema data type: int, metadata type: int64. "
            "Invalid schema found:",
        ),
        # test_varlen_feature_rank_mismatch
        (
            """{
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
                ]}""",
            {"int_list_list": VarlenFeature([1, -1, 2], tf.dtypes.int32)},
            [{"int_list_list": [[1, 2], [3]]}],
            "Mismatch between avro schema rank and metadata rank in feature 'int_list_list'. "
            "Avro schema rank: 2, metadata rank: 3. "
            "Invalid schema found:",
        ),
        # test_sparse_missing_indices_column
        (
            """{
                "type": "record",
                "name": "sparse_test",
                "fields": [
                    {
                    "name": "sparse_feature",
                    "type": {
                        "type": "record",
                        "name": "ignore_name",
                        "fields": [ {
                                "name": "indices0",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }, {
                                "name": "indices2",
                                "type": {
                                    "type": "array",
                                    "items": "int"
                                }
                            }, {
                                "name": "values",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            } ]
                        }
                    }
                ]}""",
            {"sparse_feature": SparseFeature(shape=[10, 10], dtype=tf.dtypes.int64)},
            [
                {
                    "sparse_feature": {
                        "indices0": [1, 2],
                        "indices2": [3, 4],
                        "values": [10, 11],
                    }
                }
            ],
            "Sparse schema indices should be contiguous (indices0, indices1, ...). "
            "Input data schema:",
        ),
        # test_sparse_missing_values_column
        (
            """{
                "type": "record",
                "name": "row",
                "fields": [
                    {
                        "name": "sparse_key",
                        "type": {
                            "type": "record",
                            "name": "SparseTensor",
                            "fields": [ {
                                "name": "indices0",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }, {
                                "name": "indices1",
                                "type": {
                                    "type": "array",
                                    "items": "int"
                                }
                            }]
                        }
                    }
                ]}""",
            {"sparse_key": SparseFeature(shape=[10, 10], dtype=tf.dtypes.int64)},
            [{"sparse_key": {"indices0": [1, 2], "indices1": [3, 4]}}],
            "Sparse schema is missing values column. Input data schema:",
        ),
        # test_sparse_extra_column
        (
            """{
                "type": "record",
                "name": "row",
                "fields": [
                    {
                        "name": "sparse_key",
                        "type": {
                            "type": "record",
                            "name": "SparseTensor",
                            "fields": [ {
                                "name": "indices0",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }, {
                                "name": "indices1",
                                "type": {
                                    "type": "array",
                                    "items": "int"
                                }
                            }, {
                                "name": "values",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }, {
                                "name": "extraColumn",
                                "type": {
                                    "type": "array",
                                    "items": "int"
                                }
                            }]
                        }
                    }
                ]}""",
            {"sparse_key": SparseFeature(shape=[10, 10], dtype=tf.dtypes.int64)},
            [
                {
                    "sparse_key": {
                        "indices0": [1, 2],
                        "indices1": [3, 4],
                        "values": [10, 11],
                        "extraColumn": [100, 101],
                    }
                }
            ],
            "Sparse schema can only contain 'indices' columns and a 'values' column. "
            "Input data schema:",
        ),
        # test_sparse_invalid_indices_array
        (
            """{
                "type": "record",
                "name": "row",
                "fields": [
                    {
                        "name": "sparse_key",
                        "type": {
                            "type": "record",
                            "name": "SparseTensor",
                            "fields": [ {
                                "name": "indices0",
                                "type": "int"
                            }, {
                                "name": "values",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }]
                        }
                    }
                ]}""",
            {"sparse_key": SparseFeature(shape=[10], dtype=tf.dtypes.int64)},
            [{"sparse_key": {"indices0": 1, "values": [10, 11]}}],
            "Unsupported indices type found in feature 'sparse_key'. "
            "Sparse tensor indices must be a non-nullable array of non-nullable int or long. "
            "Invalid schema found:",
        ),
        # test_sparse_invalid_indices_type
        (
            """{
                "type": "record",
                "name": "row",
                "fields": [
                    {
                    "name": "sparse_key",
                    "type": {
                        "type": "record",
                        "name": "SparseTensor",
                        "fields": [ {
                            "name": "indices0",
                            "type": {
                                "type": "array",
                                "items": "float"
                            }
                            }, {
                                "name": "values",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }]
                        }
                    }
                ]}""",
            {"sparse_key": SparseFeature(shape=[10], dtype=tf.dtypes.int64)},
            [{"sparse_key": {"indices0": [0.1, 1.1], "values": [10, 11]}}],
            "Unsupported indices type found in feature 'sparse_key'. "
            "Sparse tensor indices must be a non-nullable array of non-nullable int or long. "
            "Invalid schema found:",
        ),
        # test_sparse_invalid_nested_indices
        (
            """{
                "type": "record",
                "name": "row",
                "fields": [
                    {
                        "name": "sparse_key",
                        "type": {
                            "type": "record",
                            "name": "SparseTensor",
                            "fields": [ {
                                "name": "indices0",
                                "type": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": "long"
                                    }
                                }
                            }, {
                                "name": "values",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }]
                        }
                    }
                ]}""",
            {"sparse_key": SparseFeature(shape=[10], dtype=tf.dtypes.int64)},
            [{"sparse_key": {"indices0": [[1, 2]], "values": [10, 11]}}],
            "Unsupported indices type found in feature 'sparse_key'. "
            "Sparse tensor indices must be a non-nullable array of non-nullable int or long. "
            "Invalid schema found:",
        ),
        # test_sparse_nullable_indices
        (
            """{
                "type": "record",
                "name": "row",
                "fields": [
                    {
                        "name": "sparse_key",
                        "type": {
                            "type": "record",
                            "name": "SparseTensor",
                            "fields": [ {
                                "name": "indices0",
                                "type": {
                                    "type": "array",
                                    "items": ["null", "int"],
                                    "default": null
                                }
                            }, {
                                "name": "values",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }]
                        }
                    }
                ]}""",
            {"sparse_key": SparseFeature(shape=[10], dtype=tf.dtypes.int64)},
            [{"sparse_key": {"indices0": [1, 2], "values": [10, 11]}}],
            "Unsupported indices type found in feature 'sparse_key'. "
            "Sparse tensor indices must be a non-nullable array of non-nullable int or long. "
            "Invalid schema found:",
        ),
        # test_sparse_invalid_value_array
        (
            """{
                "type": "record",
                "name": "row",
                "fields": [
                    {
                        "name": "sparse_key",
                        "type": {
                            "type": "record",
                            "name": "SparseTensor",
                            "fields": [ {
                                "name": "indices0",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }, {
                                "name": "values",
                                "type": "long"
                            }]
                        }
                    }
                ]}""",
            {"sparse_key": SparseFeature(shape=[10], dtype=tf.dtypes.int64)},
            [{"sparse_key": {"indices0": [0, 1], "values": 1}}],
            "Unsupported value type found in feature 'sparse_key'. "
            "Tensor value must be a non-nullable array of non-nullable int, long, float, double, boolean, bytes, or string. "
            "Invalid schema found:",
        ),
        # test_sparse_invalid_value_type
        (
            """{
                "type": "record",
                "name": "row",
                "fields": [
                    {
                        "name": "sparse_key",
                        "type": {
                            "type": "record",
                            "name": "SparseTensor",
                            "fields": [ {
                                "name": "indices0",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }, {
                                "name": "values",
                                "type": {
                                    "type": "array",
                                    "items": "null"
                                }
                            }]
                        }
                    }
                ]}""",
            {"sparse_key": SparseFeature(shape=[10], dtype=tf.dtypes.int64)},
            [{"sparse_key": {"indices0": [0, 1], "values": [None, None]}}],
            "Unsupported value type found in feature 'sparse_key'. "
            "Tensor value must be a non-nullable array of non-nullable int, long, float, double, boolean, bytes, or string. "
            "Invalid schema found:",
        ),
        # test_sparse_nullable_value
        (
            """{
                "type": "record",
                "name": "row",
                "fields": [
                    {
                        "name": "sparse_key",
                        "type": {
                            "type": "record",
                            "name": "SparseTensor",
                            "fields": [ {
                                "name": "indices0",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }, {
                                "name": "values",
                                "type": {
                                    "type": "array",
                                    "items": ["null", "int"],
                                    "default": null
                                }
                            }]
                        }
                    }
                ]}""",
            {"sparse_key": SparseFeature(shape=[10], dtype=tf.dtypes.int64)},
            [{"sparse_key": {"indices0": [0, 1], "values": [1, 2]}}],
            "Unsupported value type found in feature 'sparse_key'. "
            "Tensor value must be a non-nullable array of non-nullable int, long, float, double, boolean, bytes, or string. "
            "Invalid schema found:",
        ),
        # test_sparse_nullable_indices_array
        (
            """{
                "type": "record",
                "name": "row",
                "fields": [
                    {
                        "name": "sparse_key",
                        "type": {
                            "type": "record",
                            "name": "SparseTensor",
                            "fields": [ {
                                "name": "indices0",
                                "type": ["null", {
                                    "type": "array",
                                    "items": "long"
                                }]
                            }, {
                                "name": "values",
                                "type": {
                                    "type": "array",
                                    "items": "int"
                                }
                            }]
                        }
                    }
                ]}""",
            {"sparse_key": SparseFeature(shape=[10], dtype=tf.dtypes.int32)},
            [{"sparse_key": {"indices0": [0, 1], "values": [1, 2]}}],
            "Unsupported indices type found in feature 'sparse_key'. "
            "Sparse tensor indices must be a non-nullable array of non-nullable int or long. "
            "Invalid schema found:",
        ),
        # test_sparse_nullable_values_array
        (
            """{
                "type": "record",
                "name": "row",
                "fields": [
                    {
                        "name": "sparse_key",
                        "type": {
                            "type": "record",
                            "name": "SparseTensor",
                            "fields": [ {
                                "name": "indices0",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }, {
                                "name": "values",
                                "type": ["null", {
                                    "type": "array",
                                    "items": "int"
                                }]
                            }]
                        }
                    }
                ]}""",
            {"sparse_key": SparseFeature(shape=[10], dtype=tf.dtypes.int64)},
            [{"sparse_key": {"indices0": [0, 1], "values": [1, 2]}}],
            "Unsupported value type found in feature 'sparse_key'. "
            "Tensor value must be a non-nullable array of non-nullable int, long, float, double, boolean, bytes, or string. "
            "Invalid schema found:",
        ),
        # test_sparse_invalid_nested_values
        (
            """{
                "type": "record",
                "name": "row",
                "fields": [
                    {
                        "name": "sparse_key",
                        "type": {
                            "type": "record",
                            "name": "SparseTensor",
                            "fields": [ {
                                "name": "indices0",
                                "type": {
                                    "type": "array",
                                    "items": "long"
                                }
                            }, {
                                "name": "values",
                                "type": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": "int"
                                    }
                                }
                            }]
                        }
                    }
                ]}""",
            {"sparse_key": SparseFeature(shape=[10], dtype=tf.dtypes.int64)},
            [{"sparse_key": {"indices0": [0, 1], "values": [[1, 2]]}}],
            "Unsupported value type found in feature 'sparse_key'. "
            "Tensor value must be a non-nullable array of non-nullable int, long, float, double, boolean, bytes, or string. "
            "Invalid schema found:",
        ),
        # test_sparse_value_type_mismatch
        (
            """{
            "type": "record",
            "name": "row",
            "fields": [
                {
                    "name": "sparse_key",
                    "type": {
                        "type": "record",
                        "name": "SparseTensor",
                        "fields": [ {
                            "name": "indices0",
                            "type": {
                                "type": "array",
                                "items": "int"
                            }
                        }, {
                            "name": "values",
                            "type": {
                                "type": "array",
                                "items": "int"
                            }
                        }]
                    }
                }
            ]}""",
            {"sparse_key": SparseFeature(shape=[10], dtype=tf.dtypes.int64)},
            [{"sparse_key": {"indices0": [0, 1], "values": [1, 2]}}],
            "Schema value type and metadata type mismatch in feature 'sparse_key'. "
            "Avro schema data type: int, metadata type: int64. "
            "Invalid schema found:",
        ),
    ],
)
def test_feature_schema_check(schema, features, record_data, error_message):
    """test_feature_schema_check"""
    with pytest.raises(errors.InvalidArgumentError, match=re.escape(error_message)):
        dataset = create_atds_dataset(
            writer_schema=schema,
            record_data=record_data,
            features=features,
            batch_size=1,
        )
        iterator = iter(dataset)
        next(iterator)


@pytest.mark.parametrize(
    ["record_data", "feature_name"],
    [([{"dense": None}, {"dense": None}, {"dense": None}, {"dense": None}], "dense")],
)
def test_ATDSReader_skip_block_with_null_value(record_data, feature_name):
    writer_schema = """{
          "type": "record",
          "name": "row",
          "fields": [
              {
                  "name": "dense",
                  "type": ["null", "int"]
                }
          ]}"""
    schema = avro.schema.Parse(writer_schema)
    filename = os.path.join(tempfile.gettempdir(), "test.avro")
    with open(filename, "wb") as f:
        writer = DataFileWriter(f, DatumWriter(), schema)
        for record in record_data:
            writer.append(record)
        writer.close()
    features = {
        "dense": DenseFeature([], tf.int32),
    }
    error_message = (
        f"Failed to decode feature {feature_name}. " f"Reason: Feature value is null."
    )

    def _load_dataset_inside_interleave(filename, features):
        _dataset = tf.data.Dataset.from_tensor_slices([filename])
        _dataset = _dataset.interleave(
            lambda filename: ATDSDataset(
                filenames=filename,
                batch_size=2,
                drop_remainder=True,
                features=features,
                reader_buffer_size=262144,
                shuffle_buffer_size=10000,
                num_parallel_calls=4,
            ),
            cycle_length=1,
        )
        return _dataset

    with pytest.raises(errors.InvalidArgumentError, match=error_message):
        indices = tf.data.Dataset.range(1)
        dataset = indices.interleave(
            map_func=lambda x: _load_dataset_inside_interleave(filename, features),
            cycle_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False,
        )
        iterator = iter(dataset)
        for _ in range(4):
            next(iterator)


@pytest.mark.parametrize(
    [
        "filenames",
        "batch_size",
        "reader_buffer_size",
        "shuffle_buffer_size",
        "num_parallel_calls",
        "error_message",
    ],
    [
        (
            [["file_1"], ["file_2"]],
            2,
            1024,
            1024,
            1,
            "`filenames` must be a scalar or a vector.",
        ),
        (
            ["filename"],
            0,
            1024,
            1024,
            1,
            "`batch_size` must be greater than 0 but found 0",
        ),
        (
            ["filename"],
            -1,
            1024,
            1024,
            1,
            "`batch_size` must be greater than 0 but found -1",
        ),
        (
            ["filename"],
            1,
            0,
            1024,
            1,
            "`reader_buffer_size` must be greater than 0 but found 0",
        ),
        (
            ["filename"],
            1,
            -2,
            1024,
            1,
            "`reader_buffer_size` must be greater than 0 but found -2",
        ),
        (
            ["filename"],
            1,
            1024,
            -5,
            1,
            "`shuffle_buffer_size` must be greater than or equal to 0 but found -5",
        ),
        (
            ["filename"],
            1,
            1024,
            1024,
            -2,
            "`num_parallel_calls` must be a positive integer or tf.data.AUTOTUNE, got -2",
        ),
    ],
)
def test_ATDS_dataset_creation_with_invalid_argument(
    filenames,
    batch_size,
    reader_buffer_size,
    shuffle_buffer_size,
    num_parallel_calls,
    error_message,
):
    with pytest.raises(errors.InvalidArgumentError, match=re.escape(error_message)):
        ATDSDataset(
            filenames=filenames,
            batch_size=batch_size,
            features={"x": DenseFeature([], tf.dtypes.int32)},
            reader_buffer_size=reader_buffer_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_parallel_calls=num_parallel_calls,
        )


@pytest.mark.parametrize(
    ["filenames", "batch_size", "features", "error_message"],
    [
        (
            None,
            1,
            {"x": DenseFeature([], tf.int32)},
            r"Attempt to convert a value \(None\) with an unsupported type .*",
        ),
        (
            tf.data.Dataset.from_tensor_slices(["filename"]),
            1,
            {"x": DenseFeature([], tf.int32)},
            r"Attempt to convert a value .* with an unsupported type .*",
        ),
        (
            ["filename"],
            None,
            {"x": DenseFeature([], tf.int32)},
            r"Attempt to convert a value \(None\) with an unsupported type .*",
        ),
        (
            ["filename"],
            1,
            {"featureA": ([], tf.int32)},
            r"Unknown ATDS feature configuration \(\[\], tf\.int32\)\. Only .* are supported\.",
        ),
        (
            ["filename"],
            1,
            {},
            "Features dict cannot be empty and should have at least one feature.",
        ),
        (
            ["filename"],
            1,
            None,
            r"Features can only be a dict with feature name as key and "
            r"ATDS feature configuration as value but found None\. "
            r"Available feature configuration are .*",
        ),
        (
            ["filename"],
            1,
            ([], tf.int32),
            r"Features can only be a dict with feature name as key and "
            r"ATDS feature configuration as value but found \(\[\], tf\.int32\)\. "
            r"Available feature configuration are .*",
        ),
    ],
)
def test_ATDS_dataset_creation_with_value_error(
    filenames, batch_size, features, error_message
):
    with pytest.raises(ValueError, match=error_message):
        ATDSDataset(filenames=filenames, batch_size=batch_size, features=features)


@pytest.mark.parametrize(
    [
        "feature_keys",
        "feature_types",
        "sparse_dtypes",
        "sparse_shapes",
        "output_dtypes",
        "output_shapes",
        "error_message",
    ],
    [
        (
            ["feature_1"],
            ["dense", "sparse"],
            [],
            [],
            [tf.int32],
            [[]],
            "The length of feature_keys must equal to the length of "
            "feature_types. [1 != 2]",
        ),
        (
            ["feature_1", "feature_2"],
            ["dense", "dense"],
            [],
            [],
            [tf.int32],
            [[], []],
            "The length of feature_keys must equal to the length of "
            "output_dtypes. [2 != 1]",
        ),
        (
            ["feature_1"],
            ["dense"],
            [],
            [],
            [tf.int32],
            [[], []],
            "The length of feature_keys must equal to the length of "
            "output_shapes. [1 != 2]",
        ),
        (
            ["feature_1"],
            ["dense"],
            [tf.int32],
            [],
            [tf.int32],
            [[]],
            "The length of sparse_dtypes must equal to the number of "
            "sparse features configured in feature_types. [1 != 0]",
        ),
        (
            ["feature_1"],
            ["sparse"],
            [tf.int32],
            [[1], []],
            [tf.int32],
            [[1]],
            "The length of sparse_shapes must equal to the number of "
            "sparse features configured in feature_types. [2 != 1]",
        ),
        (
            ["feature_1"],
            ["ragged"],
            [],
            [],
            [tf.int32],
            [[1]],
            "Invalid feature_type, 'ragged'. Only dense, sparse, and "
            "varlen are supported.",
        ),
    ],
)
def test_atds_dataset_invalid_attribute(
    feature_keys,
    feature_types,
    sparse_dtypes,
    sparse_shapes,
    output_dtypes,
    output_shapes,
    error_message,
):
    with pytest.raises(errors.InvalidArgumentError, match=re.escape(error_message)):
        core_ops.io_atds_dataset(
            filenames="filename",
            batch_size=1,
            drop_remainder=False,
            reader_buffer_size=1024,
            shuffle_buffer_size=0,
            num_parallel_calls=1,
            feature_keys=feature_keys,
            feature_types=feature_types,
            sparse_dtypes=sparse_dtypes,
            sparse_shapes=sparse_shapes,
            output_dtypes=output_dtypes,
            output_shapes=output_shapes,
        )


@pytest.mark.parametrize(
    ["record_data", "feature_name"],
    [
        (
            [
                {
                    "dense": 0,
                    "varlen": [1, 2],
                    "sparse": {"indices0": [0], "values": [0]},
                },
                {
                    "dense": None,
                    "varlen": [],
                    "sparse": {"indices0": [0], "values": [0]},
                },
                {
                    "dense": 0,
                    "varlen": [1, 2],
                    "sparse": {"indices0": [0], "values": [0]},
                },
            ],
            "dense",
        ),
        (
            [
                {
                    "dense": 0,
                    "varlen": None,
                    "sparse": {"indices0": [0], "values": [0]},
                },
                {
                    "dense": 1,
                    "varlen": [1, 2],
                    "sparse": {"indices0": [0], "values": [0]},
                },
                {
                    "dense": 0,
                    "varlen": [1, 2],
                    "sparse": {"indices0": [0], "values": [0]},
                },
            ],
            "varlen",
        ),
        (
            [
                {"dense": 0, "varlen": [], "sparse": {"indices0": [0], "values": [0]}},
                {"dense": 1, "varlen": [1, 2], "sparse": None},
                {
                    "dense": 0,
                    "varlen": [1, 2],
                    "sparse": {"indices0": [0], "values": [0]},
                },
            ],
            "sparse",
        ),
    ],
)
def test_ATDS_dataset_with_null_value(record_data, feature_name):
    writer_schema = """{
          "type": "record",
          "name": "row",
          "fields": [
              {
                  "name": "dense",
                  "type": ["null", "int"]},
              {
                  "name": "varlen",
                  "type": [
                      {"type": "array", "items": "int"},
                      "null"
                  ]
              },
              {
                 "name": "sparse",
                 "type" : [ {
                     "type" : "record",
                     "name" : "IntSparseTensor",
                     "fields" : [ {
                       "name" : "indices0",
                       "type" : { "type" : "array", "items" : "long" }
                     }, {
                       "name" : "values",
                       "type" : { "type" : "array", "items" : "int" }
                     } ]
                 }, "null" ]
              }
          ]}"""
    schema = avro.schema.Parse(writer_schema)
    filename = os.path.join(tempfile.gettempdir(), "test.avro")
    with open(filename, "wb") as f:
        writer = DataFileWriter(f, DatumWriter(), schema)
        for record in record_data:
            writer.append(record)
        writer.close()

    features = {
        "dense": DenseFeature([], tf.int32),
        "varlen": VarlenFeature([-1], tf.int32),
        "sparse": SparseFeature([1], tf.int32),
    }
    error_message = (
        f"Failed to decode feature {feature_name}. " f"Reason: Feature value is null."
    )
    with pytest.raises(errors.InvalidArgumentError, match=error_message):
        dataset = ATDSDataset(filename, features=features, batch_size=2)
        iterator = iter(dataset)
        next(iterator)


@pytest.mark.parametrize("shuffle_buffer_size", [0, 1, 3, 5, 10, 30, 50, 100, 200])
@pytest.mark.parametrize("batch_size", [2, 5, 10])
@pytest.mark.parametrize("num_parallel_calls", [tf.data.AUTOTUNE, 10])
def test_valid_shuffle(shuffle_buffer_size, batch_size, num_parallel_calls):
    def list_from_dataset(dataset):
        as_numpy_array = [elem["x"] for elem in list(dataset.as_numpy_iterator())]
        return list(itertools.chain(*as_numpy_array))

    data_size = 100
    writer_schema = """{
            "type": "record",
            "name": "row",
            "fields": [
                {"name": "x", "type": "int"}
            ]}"""
    schema = avro.schema.Parse(writer_schema)
    filename = os.path.join(tempfile.gettempdir(), "test.avro")
    record_data = [{"x": x} for x in range(0, data_size)]
    # Generate an avro file with 10 avro blocks.
    with open(filename, "wb") as f:
        writer = DataFileWriter(f, DatumWriter(), schema)
        for i in range(len(record_data)):
            writer.append(record_data[i])
            if (i + 1) % 10 == 0:
                writer.sync()  # Dump every 10 records into an avro block.
        writer.close()

    features = {
        "x": DenseFeature([], tf.dtypes.int32),
    }
    # Generates a list of 100 epochs and check if each dataset has a different order
    list_of_lists = []
    list_of_sets = []
    num_epochs = 100
    for i in range(0, num_epochs):
        li = list_from_dataset(
            ATDSDataset(
                filenames=filename,
                features=features,
                shuffle_buffer_size=shuffle_buffer_size,
                batch_size=batch_size,
                num_parallel_calls=num_parallel_calls,
            )
        )
        list_of_lists.append(li)
        list_of_sets.append(set(li))

    for i in range(0, num_epochs):
        for j in range(0, num_epochs):
            assert (
                list_of_sets[i] == list_of_sets[j]
            ), f"Set {list_of_sets[i]} must include the elements of {list_of_sets[j]}"
            if i != j and shuffle_buffer_size > 0:
                assert (
                    list_of_lists[i] != list_of_lists[j]
                ), f"result {list_of_lists[i]} must be shuffled, and should not be identical to expected_data {list_of_lists[j]}"
            else:
                assert (
                    list_of_lists[i] == list_of_lists[j]
                ), f"result {list_of_lists[i]} is shuffled, it should be identical to expected_data {list_of_lists[j]}"


def test_empty_sparse_buffer():
    """Tests the empty sparse buffer for dense, varlen, and sparse features."""
    data_size = 100
    writer_schema = """{
        "type": "record",
        "name": "row",
        "fields": [
            {"name": "dense", "type": "int"},
            {"name": "varlen", "type": {"type": "array", "items": "float"} },
            {
                "name": "sparse",
                "type" : {
                    "type" : "record",
                    "name" : "IntSparseTensor",
                    "fields" : [ {
                        "name" : "indices0",
                        "type" : { "type" : "array", "items" : "long" }
                    }, {
                        "name" : "values",
                        "type" : { "type" : "array", "items" : "int" }
                    } ]
                }
            }
        ]}"""
    schema = avro.schema.parse(writer_schema)
    filename = os.path.join(tempfile.gettempdir(), "test.avro")
    record_data = [
        {
            "dense": random.randint(0, 100),
            "varlen": np.random.rand(random.randint(0, 100)).tolist(),
            "sparse": {
                "indices0": [random.randint(0, 4), random.randint(4, 9)],
                "values": [2 * x, 5 * x],
            },
        }
        for x in range(0, data_size)
    ]
    # Generate an avro file with 10 avro blocks.
    with open(filename, "wb") as f:
        writer = DataFileWriter(f, DatumWriter(), schema)
        for i in range(len(record_data)):
            writer.append(record_data[i])
            if (i + 1) % 10 == 0:
                writer.sync()  # Dump every 10 records into an avro block.
        writer.close()

    features = {
        "dense": DenseFeature([], tf.int32),
        "sparse": SparseFeature([10], tf.dtypes.int32),
        "varlen": VarlenFeature([-1], tf.dtypes.float32),
    }

    # ATDSReader is parallelized along blocks.
    # This test ensures that there are enough
    # threads to gaurantee a few empty sparse buffers
    dataset = ATDSDataset(
        filenames=filename,
        features=features,
        shuffle_buffer_size=100,
        batch_size=2,
        num_parallel_calls=15,
    )

    for _ in dataset:
        pass


def test_dataset_terminate():
    writer_schema = """{
        "type": "record",
        "name": "row",
        "fields": [
            {"name": "int_value", "type": "int"}
        ]}"""
    record_data = [{"int_value": 0}, {"int_value": 1}, {"int_value": 2}]
    features = {"int_value": DenseFeature([], tf.dtypes.int32)}

    def itr(dataset):
        iter(dataset)

    dataset = create_atds_dataset(
        writer_schema=writer_schema,
        record_data=record_data,
        features=features,
        batch_size=2,
        drop_remainder=False,
    )
    # Create the internal iterator and then let it get out of scope/destroyed
    # This will fail if the destructor is waiting to delete the non-existent
    # prefetch thread.
    itr(dataset)
    itr(dataset)


class ATDSDatasetTest(AvroDatasetTestBase):
    """ATDSDatasetTest"""

    @parameterized.expand([("null"), ("deflate"), ("snappy")])
    def test_decompression(self, codec):
        data_size = 100
        data_dimension = 100
        writer_schema = """{
            "type": "record",
            "name": "row",
            "fields": [
                {
                    "name": "int_1d",
                    "type": {
                        "type": "array",
                        "items": "int"
                    }
                }
            ]}"""
        int_list = np.random.randint(
            low=-100, high=100, size=data_dimension, dtype=int
        ).tolist()
        record_data = [{"int_1d": int_list} for _ in range(0, data_size)]

        features = {
            "int_1d": DenseFeature([data_dimension], tf.dtypes.int32),
        }
        expected_data = [
            {
                "int_1d": tf.convert_to_tensor(
                    list(itertools.repeat(int_list, data_size))
                ),
            }
        ]
        dataset = create_atds_dataset(
            writer_schema=writer_schema,
            record_data=record_data,
            features=features,
            batch_size=data_size,
            codec=codec,
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)

    @parameterized.expand([("null"), ("deflate"), ("snappy")])
    def test_decompression_with_auto_tune(self, codec):
        """Test cost model and auto thread tuning."""
        data_size = 128
        data_dimension = 4096
        writer_schema = """{
            "type": "record",
            "name": "row",
            "fields": [
                {
                    "name": "int_1d",
                    "type": {
                        "type": "array",
                        "items": "int"
                    }
                }
            ]}"""
        int_list = np.ones(data_dimension, dtype=int).tolist()
        record_data = [{"int_1d": int_list} for _ in range(0, data_size)]

        features = {
            "int_1d": DenseFeature([data_dimension], tf.dtypes.int32),
        }
        dataset = create_atds_dataset(
            writer_schema=writer_schema,
            record_data=record_data,
            features=features,
            batch_size=16,
            codec=codec,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        for _ in dataset:
            pass

    def test_sparse_feature_with_various_dtypes(self):
        schema = """{
                    "type": "record",
                    "name": "row",
                    "fields": [
                        {
                           "name": "int_1d",
                           "type" : {
                               "type" : "record",
                               "name" : "IntSparseTensor",
                               "fields" : [ {
                                 "name" : "indices0",
                                 "type" : { "type" : "array", "items" : "long" }
                               }, {
                                 "name" : "values",
                                 "type" : { "type" : "array", "items" : "int" }
                               } ]
                           }
                        },
                        {
                           "name": "long_2d",
                           "type" : {
                               "type" : "record",
                               "name" : "LongSparseTensor",
                               "fields" : [ {
                                 "name" : "indices0",
                                 "type" : { "type" : "array", "items" : "long" }
                               }, {
                                 "name" : "values",
                                 "type" : { "type" : "array", "items" : "long" }
                               }, {
                                 "name" : "indices1",
                                 "type" : { "type" : "array", "items" : "long" }
                               } ]
                           }
                        },
                        {
                           "name": "float_1d",
                           "type" : {
                               "type" : "record",
                               "name" : "FloatSparseTensor",
                               "fields" : [ {
                                 "name" : "indices0",
                                 "type" : { "type" : "array", "items" : "long" }
                               }, {
                                 "name" : "values",
                                 "type" : { "type" : "array", "items" : "float" }
                               } ]
                           }
                        },
                        {
                           "name": "double_3d",
                           "type" : {
                               "type" : "record",
                               "name" : "DoubleSparseTensor",
                               "fields" : [ {
                                 "name" : "indices0",
                                 "type" : { "type" : "array", "items" : "long" }
                               }, {
                                 "name" : "values",
                                 "type" : { "type" : "array", "items" : "double" }
                               }, {
                                 "name" : "indices2",
                                 "type" : { "type" : "array", "items" : "long" }
                               }, {
                                 "name" : "indices1",
                                 "type" : { "type" : "array", "items" : "long" }
                               } ]
                           }
                        },
                        {
                           "name": "string_1d",
                           "type" : {
                               "type" : "record",
                               "name" : "StringSparseTensor",
                               "fields" : [ {
                                 "name" : "indices0",
                                 "type" : { "type" : "array", "items" : "long" }
                               }, {
                                 "name" : "values",
                                 "type" : { "type" : "array", "items" : "string" }
                               } ]
                           }
                        },
                        {
                           "name": "bytes_1d",
                           "type" : {
                               "type" : "record",
                               "name" : "ByteSparseTensor",
                               "fields" : [ {
                                 "name" : "indices0",
                                 "type" : { "type" : "array", "items" : "long" }
                               }, {
                                 "name" : "values",
                                 "type" : { "type" : "array", "items" : "bytes" }
                               } ]
                           }
                        },
                        {
                           "name": "bool_1d",
                           "type" : {
                               "type" : "record",
                               "name" : "BoolSparseTensor",
                               "fields" : [ {
                                 "name" : "indices0",
                                 "type" : { "type" : "array", "items" : "long" }
                               }, {
                                 "name" : "values",
                                 "type" : { "type" : "array", "items" : "boolean" }
                               } ]
                           }
                        }
                    ]}"""
        s1 = bytes("abc", "utf-8")
        s2 = bytes("def", "utf-8")
        s3 = bytes("ijk", "utf-8")
        s4 = bytes("lmn", "utf-8")
        s5 = bytes("opq", "utf-8")
        s6 = bytes("qrs", "utf-8")
        s7 = bytes("tuv", "utf-8")
        record_data = [
            {
                "int_1d": {"indices0": [7, 9], "values": [2, 5]},
                "long_2d": {"indices0": [0], "values": [6], "indices1": [0]},
                "float_1d": {"indices0": [0, 1], "values": [0.5, -0.5]},
                "double_3d": {
                    "indices0": [0, 0, 0],
                    "indices1": [0, 0, 0],
                    "indices2": [0, 1, 2],
                    "values": [0.5, -0.5, 1.0],
                },
                "string_1d": {"indices0": [2, 5, 8, 9], "values": ["A", "B", "C", "D"]},
                "bytes_1d": {"indices0": [2, 5, 8, 9], "values": [s1, s2, s3, s4]},
                "bool_1d": {"indices0": [100], "values": [False]},
            },
            {
                "int_1d": {"indices0": [1], "values": [1]},
                "long_2d": {"indices0": [], "values": [], "indices1": []},
                "float_1d": {"indices0": [0], "values": [9.8]},
                "double_3d": {
                    "indices0": [0, 0, 0],
                    "indices1": [0, 1, 2],
                    "indices2": [0, 0, 0],
                    "values": [6.5, -1.5, 4.0],
                },
                "string_1d": {"indices0": [2], "values": ["E"]},
                "bytes_1d": {"indices0": [2], "values": [s5]},
                "bool_1d": {"indices0": [88, 97], "values": [True, True]},
            },
            {
                "int_1d": {"indices0": [2, 4], "values": [6, 8]},
                "long_2d": {"indices0": [0, 0], "values": [7, 8], "indices1": [3, 5]},
                "float_1d": {"indices0": [], "values": []},
                "double_3d": {
                    "indices0": [0, 1, 2],
                    "indices1": [0, 0, 0],
                    "indices2": [0, 0, 0],
                    "values": [3.5, -4.5, 7.0],
                },
                "string_1d": {"indices0": [8, 9], "values": ["F", "G"]},
                "bytes_1d": {"indices0": [8, 9], "values": [s6, s7]},
                "bool_1d": {"indices0": [], "values": []},
            },
        ]
        features = {
            "int_1d": SparseFeature([10], dtype=tf.dtypes.int32),
            "long_2d": SparseFeature([1, 6], dtype=tf.dtypes.int64),
            "float_1d": SparseFeature([5], dtype=tf.dtypes.float32),
            "double_3d": SparseFeature([3, 3, 3], dtype=tf.dtypes.float64),
            "string_1d": SparseFeature([-1], dtype=tf.dtypes.string),
            "bytes_1d": SparseFeature([-1], dtype=tf.dtypes.string),
            "bool_1d": SparseFeature([101], dtype=tf.dtypes.bool),
        }
        expected_data = [
            {
                "int_1d": tf.compat.v1.SparseTensorValue(
                    indices=[
                        [0, 7],
                        [0, 9],
                        [1, 1],
                        [2, 2],
                        [2, 4],
                    ],
                    values=[2, 5, 1, 6, 8],
                    dense_shape=[3, 10],
                ),
                "long_2d": tf.compat.v1.SparseTensorValue(
                    indices=[
                        [0, 0, 0],
                        [2, 0, 3],
                        [2, 0, 5],
                    ],
                    values=np.array([6, 7, 8], dtype=np.int64),
                    dense_shape=[3, 1, 6],
                ),
                "float_1d": tf.compat.v1.SparseTensorValue(
                    indices=[
                        [0, 0],
                        [0, 1],
                        [1, 0],
                    ],
                    values=np.array([0.5, -0.5, 9.8], dtype=np.float32),
                    dense_shape=[3, 5],
                ),
                "double_3d": tf.compat.v1.SparseTensorValue(
                    indices=[
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 2],
                        [1, 0, 0, 0],
                        [1, 0, 1, 0],
                        [1, 0, 2, 0],
                        [2, 0, 0, 0],
                        [2, 1, 0, 0],
                        [2, 2, 0, 0],
                    ],
                    values=np.array(
                        [0.5, -0.5, 1.0, 6.5, -1.5, 4.0, 3.5, -4.5, 7.0],
                        dtype=np.float64,
                    ),
                    dense_shape=[3, 3, 3, 3],
                ),
                "string_1d": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 2], [0, 5], [0, 8], [0, 9], [1, 2], [2, 8], [2, 9]],
                    values=["A", "B", "C", "D", "E", "F", "G"],
                    dense_shape=[3, 10],
                ),
                "bytes_1d": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 2], [0, 5], [0, 8], [0, 9], [1, 2], [2, 8], [2, 9]],
                    values=[s1, s2, s3, s4, s5, s6, s7],
                    dense_shape=[3, 10],
                ),
                "bool_1d": tf.compat.v1.SparseTensorValue(
                    indices=[
                        [0, 100],
                        [1, 88],
                        [1, 97],
                    ],
                    values=[False, True, True],
                    dense_shape=[3, 101],
                ),
            }
        ]
        dataset = create_atds_dataset(
            writer_schema=schema,
            record_data=record_data,
            features=features,
            batch_size=3,
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)
        self.assertEqual(
            dataset.element_spec,
            {
                "int_1d": tf.SparseTensorSpec([None, 10], dtype=tf.int32),
                "long_2d": tf.SparseTensorSpec([None, 1, 6], dtype=tf.int64),
                "float_1d": tf.SparseTensorSpec([None, 5], dtype=tf.float32),
                "double_3d": tf.SparseTensorSpec([None, 3, 3, 3], dtype=tf.float64),
                "string_1d": tf.SparseTensorSpec([None, None], dtype=tf.string),
                "bytes_1d": tf.SparseTensorSpec([None, None], dtype=tf.string),
                "bool_1d": tf.SparseTensorSpec([None, 101], dtype=tf.bool),
            },
        )

    def test_sparse_with_empty_tensor(self):
        schema = """{
                    "type": "record",
                    "name": "row",
                    "fields": [
                        {
                           "name": "sparse_key",
                           "type" : {
                               "type" : "record",
                               "name" : "SparseTensor",
                               "fields" : [ {
                                 "name" : "indices0",
                                 "type" : { "type" : "array", "items" : "long" }
                               }, {
                                 "name" : "values",
                                 "type" : { "type" : "array", "items" : "float" }
                               }, {
                                 "name" : "indices1",
                                 "type" : { "type" : "array", "items" : "int" }
                               } ]
                           }
                        }
                    ]}"""
        record_data = [
            {"sparse_key": {"indices0": [], "indices1": [], "values": []}},
        ]
        features = {
            "sparse_key": SparseFeature(shape=[10, 10], dtype=tf.dtypes.float32)
        }
        expected_data = [
            {
                "sparse_key": tf.compat.v1.SparseTensorValue(
                    indices=np.array([], dtype=np.int64).reshape((0, 3)),
                    values=[],
                    dense_shape=[1, 10, 10],
                )
            }
        ]
        dataset = create_atds_dataset(
            writer_schema=schema,
            record_data=record_data,
            features=features,
            batch_size=1,
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)

    def test_batching_without_dropping_remainder(self):
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"}
              ]}"""
        record_data = [{"int_value": 0}, {"int_value": 1}, {"int_value": 2}]
        features = {"int_value": DenseFeature([], tf.dtypes.int32)}
        expected_data = [
            {"int_value": tf.convert_to_tensor([0, 1])},
            {"int_value": tf.convert_to_tensor([2])},
        ]
        dataset = create_atds_dataset(
            writer_schema=writer_schema,
            record_data=record_data,
            features=features,
            batch_size=2,
            drop_remainder=False,
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)
        self.assertEqual(
            dataset.element_spec, {"int_value": tf.TensorSpec([None], dtype=tf.int32)}
        )

    def test_batching_with_dropping_remainder(self):
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "dense", "type": "int"},
                  {"name": "varlen", "type": {"type": "array", "items": "int"} },
                  {
                      "name": "sparse",
                      "type" : {
                          "type" : "record",
                          "name" : "SparseTensor",
                              "fields" : [ {
                                  "name" : "indices0",
                                  "type" : { "type" : "array", "items" : "long" }
                              }, {
                                  "name" : "values",
                                  "type" : { "type" : "array", "items" : "int" }
                              } ]
                      }
                  }
              ]}"""
        record_data = [
            {"dense": 0, "sparse": {"indices0": [0], "values": [1]}, "varlen": [2]},
            {"dense": 1, "sparse": {"indices0": [0], "values": [2]}, "varlen": [3, 4]},
            {"dense": 2, "sparse": {"indices0": [0], "values": [3]}, "varlen": []},
        ]
        features = {
            "dense": DenseFeature([], tf.dtypes.int32),
            "sparse": SparseFeature([1], tf.dtypes.int32),
            "varlen": VarlenFeature([-1], tf.dtypes.int32),
        }
        expected_data = [
            {
                "dense": tf.convert_to_tensor([0, 1]),
                "sparse": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [1, 0]],
                    values=[1, 2],
                    dense_shape=[2, 1],
                ),
                "varlen": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [1, 0], [1, 1]],
                    values=[2, 3, 4],
                    dense_shape=[2, 2],
                ),
            },
        ]
        dataset = create_atds_dataset(
            writer_schema=writer_schema,
            record_data=record_data,
            features=features,
            batch_size=2,
            drop_remainder=True,
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)
        self.assertEqual(
            dataset.element_spec,
            {
                "dense": tf.TensorSpec([2], dtype=tf.int32),
                "sparse": tf.SparseTensorSpec([2, 1], dtype=tf.int32),
                "varlen": tf.SparseTensorSpec([2, None], dtype=tf.int32),
            },
        )

    def test_sparse_with_single_indices(self):
        schema = """{
                    "type": "record",
                    "name": "row",
                    "fields": [
                        {
                           "name": "sparse_key",
                           "type" : {
                               "type" : "record",
                               "name" : "SparseTensor",
                               "fields" : [ {
                                 "name" : "indices0",
                                 "type" : { "type" : "array", "items" : "long" }
                               }, {
                                 "name" : "values",
                                 "type" : { "type" : "array", "items" : "float" }
                               } ]
                           }
                        }
                    ]}"""
        record_data = [
            {"sparse_key": {"indices0": [0, 1], "values": [0.5, -0.5]}},
            {"sparse_key": {"indices0": [7], "values": [-1.5]}},
            {"sparse_key": {"indices0": [6, 8], "values": [1.5, -2.5]}},
        ]
        features = {"sparse_key": SparseFeature(dtype=tf.dtypes.float32, shape=[10])}
        expected_data = [
            {
                "sparse_key": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [0, 1], [1, 7], [2, 6], [2, 8]],
                    values=[0.5, -0.5, -1.5, 1.5, -2.5],
                    dense_shape=[3, 10],
                )
            }
        ]
        dataset = create_atds_dataset(
            writer_schema=schema,
            record_data=record_data,
            features=features,
            batch_size=3,
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)

    def test_sparse_with_int_indices(self):
        schema = """{
                    "type": "record",
                    "name": "row",
                    "fields": [
                        {
                           "name": "sparse_key",
                           "type" : {
                               "type" : "record",
                               "name" : "SparseTensor",
                               "fields" : [ {
                                 "name" : "indices0",
                                 "type" : { "type" : "array", "items" : "long" }
                               }, {
                                 "name" : "values",
                                 "type" : { "type" : "array", "items" : "float" }
                               }, {
                                 "name" : "indices1",
                                 "type" : { "type" : "array", "items" : "int" }
                               } ]
                           }
                        }
                    ]}"""
        record_data = [
            {
                "sparse_key": {
                    "indices0": [0, 0],
                    "indices1": [1, 2],
                    "values": [0.5, -0.5],
                }
            },
            {"sparse_key": {"indices0": [7], "indices1": [0], "values": [-1.5]}},
            {
                "sparse_key": {
                    "indices0": [6, 8],
                    "indices1": [9, 2],
                    "values": [1.5, -2.5],
                }
            },
        ]
        features = {
            "sparse_key": SparseFeature(dtype=tf.dtypes.float32, shape=[10, 10])
        }
        expected_data = [
            {
                "sparse_key": tf.compat.v1.SparseTensorValue(
                    indices=[
                        [0, 0, 1],
                        [0, 0, 2],
                        [1, 7, 0],
                        [2, 6, 9],
                        [2, 8, 2],
                    ],
                    values=[0.5, -0.5, -1.5, 1.5, -2.5],
                    dense_shape=[3, 10, 10],
                )
            }
        ]
        dataset = create_atds_dataset(
            writer_schema=schema,
            record_data=record_data,
            features=features,
            batch_size=3,
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)

    def test_dense_feature_with_various_dtype(self):
        schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_1d",
                     "type": {
                        "type": "array",
                        "items": "int"
                     }
                  },
                  {
                     "name": "long_0d",
                     "type": "long"
                  },
                  {
                     "name": "float_1d",
                     "type": {
                        "type": "array",
                        "items": "float"
                     }
                  },
                  {
                     "name": "double_3d",
                     "type": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": "double"
                            }
                        }
                     }
                  },
                  {
                     "name": "string_2d",
                     "type": {
                         "type": "array",
                         "items": {
                             "type": "array",
                             "items": "string"
                         }
                     }
                  },
                  {
                     "name": "bytes_0d",
                     "type": "bytes"
                  },
                  {
                     "name": "bytes_2d",
                     "type": {
                         "type": "array",
                         "items": {
                             "type": "array",
                             "items": "bytes"
                         }
                     }
                  },
                  {
                     "name": "bool_0d",
                     "type": "boolean"
                  }
              ]}"""
        s1 = bytes("abc", "utf-8")
        s2 = bytes("def", "utf-8")
        s3 = bytes("ijk", "utf-8")

        record_data = [
            {
                "int_1d": [0, 1, 2],
                "long_0d": 7,
                "float_1d": [0.1],
                "double_3d": [[[0.9], [0.8]]],
                "string_2d": [["abc"], ["de"]],
                "bytes_0d": s1,
                "bytes_2d": [[s1], [s2]],
                "bool_0d": False,
            },
            {
                "int_1d": [3, 4, 5],
                "long_0d": 8,
                "float_1d": [0.2],
                "double_3d": [[[-0.9], [-0.8]]],
                "string_2d": [["XX"], ["YZ"]],
                "bytes_0d": s2,
                "bytes_2d": [[s2], [s3]],
                "bool_0d": True,
            },
            {
                "int_1d": [6, 7, 8],
                "long_0d": 9,
                "float_1d": [0.3],
                "double_3d": [[[1.5e10], [1.1e20]]],
                "string_2d": [["CK"], [""]],
                "bytes_0d": s3,
                "bytes_2d": [[s3], [s1]],
                "bool_0d": False,
            },
        ]
        features = {
            "int_1d": DenseFeature([3], tf.dtypes.int32),
            "long_0d": DenseFeature([], tf.dtypes.int64),
            "float_1d": DenseFeature([1], tf.dtypes.float32),
            "double_3d": DenseFeature([1, 2, 1], tf.dtypes.float64),
            "string_2d": DenseFeature([2, 1], tf.dtypes.string),
            "bytes_0d": DenseFeature([], tf.dtypes.string),
            "bytes_2d": DenseFeature([2, 1], tf.dtypes.string),
            "bool_0d": DenseFeature([], tf.dtypes.bool),
        }
        expected_data = [
            {
                "int_1d": tf.convert_to_tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                "long_0d": tf.convert_to_tensor(np.array([7, 8, 9], dtype=np.int64)),
                "float_1d": tf.convert_to_tensor(
                    np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
                ),
                "double_3d": tf.convert_to_tensor(
                    np.array(
                        [[[[0.9], [0.8]]], [[[-0.9], [-0.8]]], [[[1.5e10], [1.1e20]]]],
                        dtype=np.float64,
                    )
                ),
                "string_2d": tf.convert_to_tensor(
                    [[["abc"], ["de"]], [["XX"], ["YZ"]], [["CK"], [""]]]
                ),
                "bytes_0d": tf.convert_to_tensor([s1, s2, s3]),
                "bytes_2d": tf.convert_to_tensor(
                    [[[s1], [s2]], [[s2], [s3]], [[s3], [s1]]]
                ),
                "bool_0d": tf.convert_to_tensor([False, True, False]),
            },
        ]

        dataset = create_atds_dataset(
            writer_schema=schema,
            record_data=record_data,
            features=features,
            batch_size=3,
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)
        self.assertEqual(
            dataset.element_spec,
            {
                "int_1d": tf.TensorSpec([None, 3], dtype=tf.int32),
                "long_0d": tf.TensorSpec([None], dtype=tf.int64),
                "float_1d": tf.TensorSpec([None, 1], dtype=tf.float32),
                "double_3d": tf.TensorSpec([None, 1, 2, 1], dtype=tf.float64),
                "string_2d": tf.TensorSpec([None, 2, 1], dtype=tf.string),
                "bytes_0d": tf.TensorSpec([None], dtype=tf.string),
                "bytes_2d": tf.TensorSpec([None, 2, 1], dtype=tf.string),
                "bool_0d": tf.TensorSpec([None], dtype=tf.bool),
            },
        )

    def test_skipping_opaque_contextual_columns(self):
        schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "opaque_contextual_column_1",
                     "type": {
                        "type": "array",
                        "items": "int"
                     }
                  },
                  {
                     "name": "opaque_contextual_column_3",
                     "type": "string"
                  },
                  {
                     "name": "feature",
                     "type": {
                        "type": "array",
                        "items": "float"
                     }
                  },
                  {
                     "name": "opaque_contextual_column_2",
                     "type" : {
                         "type" : "record",
                         "name" : "TermValues",
                         "fields" : [ {
                           "name" : "term",
                           "type" : { "type" : "array", "items" : "string" }
                         }, {
                           "name" : "values",
                           "type" : { "type" : "array", "items" : "float" }
                         } ]
                     }
                  }
        ]}"""
        record_data = [
            {
                "opaque_contextual_column_1": [0, 1, 2],
                "feature": [0.1],
                "opaque_contextual_column_3": "ABC",
                "opaque_contextual_column_2": {
                    "term": ["A", "B"],
                    "values": [0.5, -0.5],
                },
            },
            {
                "opaque_contextual_column_1": [],
                "feature": [0.2],
                "opaque_contextual_column_3": "DEF",
                "opaque_contextual_column_2": {"term": ["C"], "values": [1.0]},
            },
            {
                "opaque_contextual_column_1": [135],
                "feature": [0.3],
                "opaque_contextual_column_3": "GH",
                "opaque_contextual_column_2": {"term": [], "values": [1.8]},
            },
            {
                "opaque_contextual_column_1": [-2, -3],
                "feature": [0.4],
                "opaque_contextual_column_3": "I",
                "opaque_contextual_column_2": {
                    "term": ["A", "B", "C"],
                    "values": [0.5],
                },
            },
        ]
        features = {
            "feature": DenseFeature([1], tf.dtypes.float32),
        }
        expected_data = [
            {
                "feature": tf.convert_to_tensor(
                    np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
                ),
            },
        ]

        dataset = create_atds_dataset(
            writer_schema=schema,
            record_data=record_data,
            features=features,
            batch_size=4,
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)

    def test_varlen_feature_with_various_dtypes(self):
        schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "int_feature",
                     "type": {
                        "type": "array",
                        "items": "int"
                     }
                  },
                  {
                     "name": "long_feature",
                     "type": "long"
                  },
                  {
                     "name": "float_feature",
                     "type": {
                        "type": "array",
                        "items": "float"
                     }
                  },
                  {
                     "name": "double_feature",
                     "type": {
                         "type": "array",
                         "items": {
                             "type": "array",
                             "items": "double"
                         }
                     }
                  },
                  {
                     "name": "string_feature",
                     "type": {
                         "type": "array",
                         "items": {
                             "type": "array",
                             "items": "string"
                         }
                     }
                  },
                  {
                     "name": "bytes_feature",
                     "type": {
                         "type": "array",
                         "items": {
                             "type": "array",
                             "items": "bytes"
                         }
                     }
                  },
                  {
                     "name": "bool_feature",
                     "type": {
                         "type": "array",
                         "items": {
                             "type": "array",
                             "items": {
                                 "type": "array",
                                 "items": "boolean"
                             }
                         }
                     }
                  }
              ]}"""
        s1 = bytes("abc", "utf-8")
        s2 = bytes("def", "utf-8")
        s3 = bytes("ijk", "utf-8")
        s4 = bytes("lmn", "utf-8")
        s5 = bytes("opq", "utf-8")
        s6 = bytes("qrs", "utf-8")
        record_data = [
            {
                "int_feature": [0],
                "long_feature": 1,
                "float_feature": [1.5, -2.7],
                "double_feature": [[3.9], [-1.0, 1.0]],
                "string_feature": [["abc"], ["de"]],
                "bytes_feature": [[s1], [s2]],
                "bool_feature": [[[True]], [[False, False], [True]]],
            },
            {
                "int_feature": [],
                "long_feature": -1,
                "float_feature": [2.0, 3.0],
                "double_feature": [[], [7.0]],
                "string_feature": [["fg"], ["hi"], ["jk"]],
                "bytes_feature": [[s3], [s4], [s5]],
                "bool_feature": [[[False]], [[False, True, True]]],
            },
            {
                "int_feature": [1, 2],
                "long_feature": 2,
                "float_feature": [5.5, 6.5],
                "double_feature": [[], []],
                "string_feature": [["lmn"]],
                "bytes_feature": [[s6]],
                "bool_feature": [[[True], [False]]],
            },
        ]
        features = {
            "int_feature": VarlenFeature([-1], tf.dtypes.int32),
            "long_feature": VarlenFeature([], tf.dtypes.int64),
            "float_feature": VarlenFeature([2], tf.dtypes.float32),
            "double_feature": VarlenFeature([-1, -1], tf.dtypes.float64),
            "string_feature": VarlenFeature([-1, 1], tf.dtypes.string),
            "bytes_feature": VarlenFeature([-1, 1], tf.dtypes.string),
            "bool_feature": VarlenFeature([-1, -1, -1], tf.dtypes.bool),
        }
        expected_data = [
            {
                "int_feature": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [2, 0], [2, 1]],
                    values=[0, 1, 2],
                    dense_shape=[3, 2],
                ),
                "long_feature": tf.compat.v1.SparseTensorValue(
                    indices=[[0], [1], [2]],
                    values=np.array([1, -1, 2], dtype=np.int64),
                    dense_shape=[3],
                ),
                "float_feature": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]],
                    values=np.array([1.5, -2.7, 2.0, 3.0, 5.5, 6.5], dtype=np.float32),
                    dense_shape=[3, 2],
                ),
                "double_feature": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0]],
                    values=np.array([3.9, -1.0, 1.0, 7.0], dtype=np.float64),
                    dense_shape=[3, 2, 2],
                ),
                "string_feature": tf.compat.v1.SparseTensorValue(
                    indices=[
                        [0, 0, 0],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 1, 0],
                        [1, 2, 0],
                        [2, 0, 0],
                    ],
                    values=["abc", "de", "fg", "hi", "jk", "lmn"],
                    dense_shape=[3, 3, 1],
                ),
                "bytes_feature": tf.compat.v1.SparseTensorValue(
                    indices=[
                        [0, 0, 0],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 1, 0],
                        [1, 2, 0],
                        [2, 0, 0],
                    ],
                    values=[s1, s2, s3, s4, s5, s6],
                    dense_shape=[3, 3, 1],
                ),
                "bool_feature": tf.compat.v1.SparseTensorValue(
                    indices=[
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 1],
                        [0, 1, 1, 0],
                        [1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 0, 1],
                        [1, 1, 0, 2],
                        [2, 0, 0, 0],
                        [2, 0, 1, 0],
                    ],
                    values=[
                        True,
                        False,
                        False,
                        True,
                        False,
                        False,
                        True,
                        True,
                        True,
                        False,
                    ],
                    dense_shape=[3, 2, 2, 3],
                ),
            },
        ]

        dataset = create_atds_dataset(
            writer_schema=schema,
            record_data=record_data,
            features=features,
            batch_size=3,
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)
        self.assertEqual(
            dataset.element_spec,
            {
                "int_feature": tf.SparseTensorSpec([None, None], dtype=tf.int32),
                "long_feature": tf.SparseTensorSpec([None], dtype=tf.int64),
                "float_feature": tf.SparseTensorSpec([None, 2], dtype=tf.float32),
                "double_feature": tf.SparseTensorSpec(
                    [None, None, None], dtype=tf.float64
                ),
                "string_feature": tf.SparseTensorSpec([None, None, 1], dtype=tf.string),
                "bytes_feature": tf.SparseTensorSpec([None, None, 1], dtype=tf.string),
                "bool_feature": tf.SparseTensorSpec(
                    [None, None, None, None], dtype=tf.bool
                ),
            },
        )

    def test_sparse_feature_serialization_deserialization(self):
        schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {
                     "name": "x",
                     "type": {
                        "type": "array",
                        "items": "int"
                     }
                  }
              ]}"""
        record_data = [
            {"x": [0]},
            {"x": []},
            {"x": [1, 2]},
        ]
        features = {
            "x": VarlenFeature([-1], tf.dtypes.int32),
        }
        expected_data = [
            {
                "x": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0]],
                    values=[0],
                    dense_shape=[2, 1],
                ),
            },
            {
                "x": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [0, 1]],
                    values=[1, 2],
                    dense_shape=[1, 2],
                ),
            },
        ]

        dataset = create_atds_dataset(
            writer_schema=schema,
            record_data=record_data,
            features=features,
            batch_size=2,
        )
        dataset = dataset.map(lambda d: {"x": tf.io.serialize_many_sparse(d["x"])})
        dataset = dataset.map(
            lambda d: {"x": tf.io.deserialize_many_sparse(d["x"], dtype=tf.int32)}
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)

    def test_ATDS_dataset_with_interleave(self):
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"}
              ]}"""
        record_data = [{"int_value": 0}, {"int_value": 1}, {"int_value": 2}]
        features = {"int_value": DenseFeature([], tf.dtypes.int32)}
        expected_data = [
            {"int_value": tf.convert_to_tensor([0, 1])},
            {"int_value": tf.convert_to_tensor([2])},
        ]
        filenames = AvroDatasetTestBase._setup_files(
            writer_schema=writer_schema, records=record_data
        )
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.interleave(
            lambda x: ATDSDataset(x, features=features, batch_size=2)
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)

    def test_ATDS_dataset_with_file_not_existed(self):
        filename = "file_not_exist"
        error_message = f".*{filename}.*"
        with pytest.raises(errors.NotFoundError, match=error_message):
            dataset = ATDSDataset(
                filename, features={"x": DenseFeature([], tf.int32)}, batch_size=2
            )
            iterator = iter(dataset)
            next(iterator)

    def test_ATDS_dataset_with_feature_not_existed(self):
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "int_value", "type": "int"}
              ]}"""
        record_data = [{"int_value": 0}, {"int_value": 1}, {"int_value": 2}]
        filenames = AvroDatasetTestBase._setup_files(
            writer_schema=writer_schema, records=record_data
        )

        feature_name = "feature_not_existed"
        features = {feature_name: DenseFeature([], tf.dtypes.int32)}
        error_message = (
            f"User defined feature '{feature_name}' cannot be found"
            f" in the input data. Input data schema: .*"
        )
        with pytest.raises(errors.InvalidArgumentError, match=error_message):
            dataset = ATDSDataset(filenames, features=features, batch_size=2)
            iterator = iter(dataset)
            next(iterator)

    def test_ATDS_dataset_with_null_schema(self):
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "dense_0d", "type": ["null", "int"]},
                  {
                      "name": "dense_1d",
                      "type": [
                          {"type": "array", "items": "int"},
                          "null"
                      ]
                  },
                  {
                     "name": "sparse",
                     "type" : [ {
                         "type" : "record",
                         "name" : "IntSparseTensor",
                         "fields" : [ {
                           "name" : "indices0",
                           "type" : { "type" : "array", "items" : "long" }
                         }, {
                           "name" : "values",
                           "type" : { "type" : "array", "items" : "int" }
                         } ]
                     }, "null" ]
                  },
                  {"name": "non_null", "type": "int"}
              ]}"""
        record_data = [
            {
                "dense_0d": 0,
                "dense_1d": [1],
                "non_null": 1,
                "sparse": {"indices0": [0], "values": [1]},
            },
            {
                "dense_0d": 1,
                "dense_1d": [2],
                "non_null": 2,
                "sparse": {"indices0": [0], "values": [2]},
            },
            {
                "dense_0d": 2,
                "dense_1d": [3],
                "non_null": 3,
                "sparse": {"indices0": [0], "values": [3]},
            },
        ]
        features = {
            "dense_0d": DenseFeature([], tf.int32),
            "dense_1d": DenseFeature([1], tf.int32),
            "sparse": SparseFeature([1], tf.int32),
            "non_null": DenseFeature([], tf.int32),
        }
        expected_data = [
            {
                "dense_0d": tf.convert_to_tensor([0, 1, 2]),
                "dense_1d": tf.convert_to_tensor([[1], [2], [3]]),
                "sparse": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [1, 0], [2, 0]],
                    values=[1, 2, 3],
                    dense_shape=[3, 1],
                ),
                "non_null": tf.convert_to_tensor([1, 2, 3]),
            }
        ]

        dataset = create_atds_dataset(
            writer_schema=writer_schema,
            record_data=record_data,
            features=features,
            batch_size=3,
        )
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)

    def test_ATDS_dataset_with_multithreading(self):
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "dense", "type": {"type": "array", "items": "int"}},
                  {"name": "varlen", "type": {"type": "array", "items": "int"} },
                  {
                      "name": "sparse",
                      "type" : {
                          "type" : "record",
                          "name" : "IntSparseTensor",
                          "fields" : [ {
                              "name" : "indices0",
                              "type" : { "type" : "array", "items" : "long" }
                          }, {
                              "name" : "values",
                              "type" : { "type" : "array", "items" : "int" }
                          } ]
                      }
                  }
              ]}"""
        schema = avro.schema.Parse(writer_schema)
        filename = os.path.join(tempfile.gettempdir(), "test.avro")
        record_data = [
            {
                "dense": [0, 1, 2],
                "sparse": {"indices0": [0], "values": [1]},
                "varlen": [2],
            },
            {
                "dense": [3, 4, 5],
                "sparse": {"indices0": [1], "values": [2]},
                "varlen": [3],
            },
            {
                "dense": [6, 7, 8],
                "sparse": {"indices0": [2], "values": [3]},
                "varlen": [],
            },
            {
                "dense": [9, 10, 11],
                "sparse": {"indices0": [3], "values": [10]},
                "varlen": [5],
            },
            {
                "dense": [12, 13, 14],
                "sparse": {"indices0": [4], "values": [1000]},
                "varlen": [6, 7, 8],
            },
        ]

        # Generate an avro file with 5 avro blocks.
        with open(filename, "wb") as f:
            writer = DataFileWriter(f, DatumWriter(), schema)
            for record in record_data:
                writer.append(record)
                writer.sync()  # Dump the current record into an avro block.
            writer.close()

        features = {
            "dense": DenseFeature([3], tf.dtypes.int32),
            "sparse": SparseFeature([1], tf.dtypes.int32),
            "varlen": VarlenFeature([-1], tf.dtypes.int32),
        }

        expected_data = [
            {
                "dense": tf.convert_to_tensor(
                    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
                ),
                "sparse": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
                    values=[1, 2, 3, 10, 1000],
                    dense_shape=[5, 1],
                ),
                "varlen": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [1, 0], [3, 0], [4, 0], [4, 1], [4, 2]],
                    values=[2, 3, 5, 6, 7, 8],
                    dense_shape=[5, 3],
                ),
            }
        ]
        dataset = ATDSDataset(
            filenames=filename,
            features=features,
            batch_size=5,
            num_parallel_calls=3,  # Process 5 blocks with 3 threads concurrently
        )
        # Result should have deterministic order.
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)

    def test_ATDS_dataset_processing_multiple_files(self):
        writer_schema = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "dense", "type": "int"},
                  {"name": "varlen", "type": {"type": "array", "items": "float"} },
                  {
                      "name": "sparse",
                      "type" : {
                          "type" : "record",
                          "name" : "SparseTensor",
                              "fields" : [ {
                                  "name" : "indices0",
                                  "type" : { "type" : "array", "items" : "long" }
                              }, {
                                  "name" : "values",
                                  "type" : { "type" : "array", "items" : "long" }
                              } ]
                      }
                  }
              ]}"""
        record_data = [
            {"dense": 0, "sparse": {"indices0": [0], "values": [1]}, "varlen": [2.0]},
            {
                "dense": 1,
                "sparse": {"indices0": [1], "values": [2]},
                "varlen": [3.0, 4.0],
            },
            {"dense": 2, "sparse": {"indices0": [2], "values": [3]}, "varlen": []},
            {"dense": 3, "sparse": {"indices0": [3], "values": [4]}, "varlen": [5.0]},
            {
                "dense": 4,
                "sparse": {"indices0": [4], "values": [5]},
                "varlen": [6.0, 7.0, 8.0],
            },
            {"dense": 5, "sparse": {"indices0": [5], "values": [6]}, "varlen": [9.0]},
        ]

        schema = avro.schema.Parse(writer_schema)
        # Generate 3 avro files with 2 records in each file.
        temp_dir = tempfile.gettempdir()
        filenames = []
        for i in range(3):
            filename = os.path.join(temp_dir, f"test-{i}.avro")
            with open(filename, "wb") as f:
                writer = DataFileWriter(f, DatumWriter(), schema)
                for r in range(2):
                    writer.append(record_data[i * 2 + r])
                writer.close()
            filenames.append(filename)

        features = {
            "dense": DenseFeature([], tf.dtypes.int32),
            "sparse": SparseFeature([10], tf.dtypes.int64),
            "varlen": VarlenFeature([-1], tf.dtypes.float32),
        }
        expected_data = [
            {
                "dense": tf.convert_to_tensor([0, 1, 2]),
                "sparse": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [1, 1], [2, 2]],
                    values=[1, 2, 3],
                    dense_shape=[3, 10],
                ),
                "varlen": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [1, 0], [1, 1]],
                    values=[2.0, 3.0, 4.0],
                    dense_shape=[3, 2],
                ),
            },
            {
                "dense": tf.convert_to_tensor([3, 4, 5]),
                "sparse": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 3], [1, 4], [2, 5]],
                    values=[4, 5, 6],
                    dense_shape=[3, 10],
                ),
                "varlen": tf.compat.v1.SparseTensorValue(
                    indices=[[0, 0], [1, 0], [1, 1], [1, 2], [2, 0]],
                    values=[5.0, 6.0, 7.0, 8.0, 9.0],
                    dense_shape=[3, 3],
                ),
            },
        ]
        dataset = ATDSDataset(filenames=filenames, features=features, batch_size=3)
        # Result should have deterministic order.
        self._verify_output(expected_data=expected_data, actual_dataset=dataset)

    def test_ATDS_dataset_processing_multiple_files_with_different_schema(self):
        # Generate 2 avro files with different schema.
        writer_schema_1 = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "dense", "type": "int"},
                  {"name": "varlen", "type": {"type": "array", "items": "float"} },
                  {
                      "name": "sparse",
                      "type" : {
                          "type" : "record",
                          "name" : "SparseTensor",
                              "fields" : [ {
                                  "name" : "indices0",
                                  "type" : { "type" : "array", "items" : "long" }
                              }, {
                                  "name" : "values",
                                  "type" : { "type" : "array", "items" : "long" }
                              } ]
                      }
                  }
              ]}"""
        record_data_1 = [
            {"dense": 0, "sparse": {"indices0": [0], "values": [1]}, "varlen": [2.0]},
            {
                "dense": 1,
                "sparse": {"indices0": [1], "values": [2]},
                "varlen": [3.0, 4.0],
            },
            {"dense": 2, "sparse": {"indices0": [2], "values": [3]}, "varlen": []},
        ]
        filenames_1 = AvroDatasetTestBase._setup_files(
            writer_schema=writer_schema_1, records=record_data_1
        )

        writer_schema_2 = """{
              "type": "record",
              "name": "row",
              "fields": [
                  {"name": "dense", "type": "int"},
                  {"name": "varlen", "type": {"type": "array", "items": "float"} },
                  {"name": "unused", "type": {"type": "array", "items": "float"} },
                  {
                      "name": "sparse",
                      "type" : {
                          "type" : "record",
                          "name" : "SparseTensor",
                              "fields" : [ {
                                  "name" : "indices0",
                                  "type" : { "type" : "array", "items" : "int" }
                              }, {
                                  "name" : "values",
                                  "type" : { "type" : "array", "items" : "long" }
                              } ]
                      }
                  }
              ]}"""
        record_data_2 = [
            {
                "dense": 3,
                "sparse": {"indices0": [3], "values": [4]},
                "varlen": [5.0],
                "unused": [1.0],
            },
            {
                "dense": 4,
                "sparse": {"indices0": [4], "values": [5]},
                "varlen": [6.0, 7.0, 8.0],
                "unused": [],
            },
            {
                "dense": 5,
                "sparse": {"indices0": [5], "values": [6]},
                "varlen": [9.0],
                "unused": [-1.0, 2.0],
            },
        ]
        filenames_2 = AvroDatasetTestBase._setup_files(
            writer_schema=writer_schema_2, records=record_data_2
        )
        filenames = filenames_1 + filenames_2

        features = {
            "dense": DenseFeature([], tf.dtypes.int32),
            "sparse": SparseFeature([10], tf.dtypes.int64),
            "varlen": VarlenFeature([-1], tf.dtypes.float32),
        }
        error_message = (
            "Avro schema should be consistent for all input files. "
            "Schema in file .* varies from the schema in file .*"
        )
        with pytest.raises(errors.InvalidArgumentError, match=error_message):
            dataset = ATDSDataset(filenames=filenames, features=features, batch_size=3)
            iterator = iter(dataset)
            next(iterator)  # load first file
            next(iterator)  # load second file
