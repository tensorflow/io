# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""ATDS benchmark for schema with mixed data types."""

import glob
import os
import pytest
import tensorflow as tf

from tests.test_atds_avro.utils.data_source import DataSource
from tests.test_atds_avro.utils.data_source_registry import SMALL_NUM_RECORDS
from tests.test_atds_avro.utils.generator.tensor_generator import (
    IntTensorGenerator,
    FloatTensorGenerator,
    WordTensorGenerator,
)
from tests.test_atds_avro.utils.generator.sparse_tensor_generator import (
    FloatSparseTensorGenerator,
    ValueDistribution,
)
from tests.test_atds_avro.utils.atds_writer import ATDSWriter
from tests.test_atds_avro.utils.benchmark_utils import benchmark_func
from tests.test_atds_avro.utils.atds_benchmark_utils import (
    get_dataset,
    get_features_from_data_source,
)


@pytest.mark.benchmark(
    group="mixed",
)
def test_mixed_benchmark_data():
    scenario = {
        "sparse_1d_float_small_1": FloatSparseTensorGenerator(
            tf.SparseTensorSpec([3], tf.dtypes.float32), ValueDistribution.SINGLE_VALUE
        ),
        "sparse_1d_float_large": FloatSparseTensorGenerator(
            tf.SparseTensorSpec([50001], tf.dtypes.float32),
            ValueDistribution.SINGLE_VALUE,
        ),
        "dense_0d_float": FloatTensorGenerator(tf.TensorSpec([], tf.dtypes.float32)),
        "dense_1d_float_large_1": FloatTensorGenerator(
            tf.TensorSpec([200], tf.dtypes.float32)
        ),
        "dense_0d_int_1": IntTensorGenerator(tf.TensorSpec([], tf.dtypes.int32)),
        "sparse_1d_float_medium_1": FloatSparseTensorGenerator(
            tf.SparseTensorSpec([10], tf.dtypes.float32), ValueDistribution.SINGLE_VALUE
        ),
        "dense_1d_float_large_2": FloatTensorGenerator(
            tf.TensorSpec([200], tf.dtypes.float32)
        ),
        "dense_1d_float_small_1": FloatTensorGenerator(
            tf.TensorSpec([2], tf.dtypes.float32)
        ),
        "dense_1d_float_large_3": FloatTensorGenerator(
            tf.TensorSpec([200], tf.dtypes.float32)
        ),
        "dense_1d_float_small_2": FloatTensorGenerator(
            tf.TensorSpec([2], tf.dtypes.float32)
        ),
        "dense_1d_float_small_3": FloatTensorGenerator(
            tf.TensorSpec([2], tf.dtypes.float32)
        ),
        "sparse_1d_float_medium_2": FloatSparseTensorGenerator(
            tf.SparseTensorSpec([51], tf.dtypes.float32), ValueDistribution.SINGLE_VALUE
        ),
        "sparse_1d_float_small_2": FloatSparseTensorGenerator(
            tf.SparseTensorSpec([3], tf.dtypes.float32), ValueDistribution.SINGLE_VALUE
        ),
        "dense_1d_float_large_4": FloatTensorGenerator(
            tf.TensorSpec([200], tf.dtypes.float32)
        ),
        "dense_1d_float_small_4": FloatTensorGenerator(
            tf.TensorSpec([1], tf.dtypes.float32)
        ),
        "dense_0d_string_1": WordTensorGenerator(
            tf.TensorSpec([], tf.dtypes.string), avg_length=24
        ),
        "dense_0d_int_2": IntTensorGenerator(tf.TensorSpec([], tf.dtypes.int32)),
        "dense_0d_string_2": WordTensorGenerator(
            tf.TensorSpec([], tf.dtypes.string), avg_length=24
        ),
        "dense_0d_long": IntTensorGenerator(tf.TensorSpec([], tf.dtypes.int64)),
    }
    num_partitions = 10
    data_source = DataSource(
        scenario=scenario, num_records=SMALL_NUM_RECORDS, partitions=num_partitions
    )
    with ATDSWriter() as writer:
        dir_path = writer.write(data_source)
        pattern = os.path.join(dir_path, f"*.{writer.extension}")
        dataset = get_dataset(
            glob.glob(pattern), get_features_from_data_source(writer, data_source)
        )
        dataset = dataset.unbatch()
        benchmark_func(dataset)
