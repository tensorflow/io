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
"""Utility functions for benchmarks."""

import os
import tensorflow as tf

from tests.test_atds_avro.utils.data_source_registry import (
    get_canonical_name,
    get_data_source_from_registry,
)
from tests.test_atds_avro.utils.generator.tensor_generator import (
    IntTensorGenerator,
    FloatTensorGenerator,
    BoolTensorGenerator,
)
from tests.test_atds_avro.utils.generator.sparse_tensor_generator import (
    IntSparseTensorGenerator,
    ValueDistribution,
)
from tests.test_atds_avro.utils.generator.varlen_tensor_generator import (
    WordVarLenTensorGenerator,
    DimensionDistribution,
)
from tests.test_atds_avro.utils.tf_record_writer import TFRecordWriter

MIXED_TYPES_SCENARIO = {
    # simulate scalar int as label.
    "int32_0d_dense": IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int32)),
    # simulate large sparse categorical ids.
    "int64_1d_sparse": IntSparseTensorGenerator(
        tf.SparseTensorSpec(shape=[50000], dtype=tf.int32),
        ValueDistribution.SINGLE_VALUE,
    ),
    # simulate 1d float embedding input.
    "float32_1d_varlen": FloatTensorGenerator(
        tf.TensorSpec(shape=[128], dtype=tf.float32)
    ),
    # simulate 2d images
    "float64_2d_dense": FloatTensorGenerator(
        tf.TensorSpec(shape=[32, 32], dtype=tf.float64)
    ),
    # simulate a sentence with varlen words.
    "string_1d_sparse": WordVarLenTensorGenerator(
        tf.SparseTensorSpec(shape=[None], dtype=tf.string),
        DimensionDistribution.LARGE_DIM,
    ),
    # simulate concatenated bool wide features.
    "bool_1d_dense": BoolTensorGenerator(tf.TensorSpec(shape=[5], dtype=tf.bool)),
}


def benchmark_func(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count


def create_tf_record_dataset(
    filenames, parse_function, batch_size, file_parallelism=None, shuffle_buffer_size=0
):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=file_parallelism)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_function)
    dataset = dataset.prefetch(1)
    return dataset


def run_tf_record_benchmark(
    tensor_type, rank, dtype, num_records, partitions, batch_size, benchmark
):
    data_source_name = get_canonical_name(
        tensor_type, rank, dtype, num_records, partitions
    )
    data_source = get_data_source_from_registry(data_source_name)
    run_tf_record_benchmark_from_data_source(data_source, batch_size, benchmark)


def run_tf_record_benchmark_from_data_source(
    data_source,
    batch_size,
    benchmark,
    file_parallelism=None,
    shuffle_buffer_size=0,
    rounds=100,
):
    with TFRecordWriter() as writer:
        dir_path = writer.write(data_source)
        pattern = os.path.join(dir_path, f"*.{writer.extension}")
        filenames = tf.data.Dataset.list_files(pattern)
        parse_function = writer.create_tf_example_parser_fn(
            data_source, with_batch=True
        )
        dataset = create_tf_record_dataset(
            filenames,
            parse_function,
            batch_size,
            file_parallelism=file_parallelism,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        count = benchmark.pedantic(
            target=benchmark_func,
            args=[dataset],
            iterations=2,
            # pytest-benchmark calculates statistic across rounds. Set it with
            # larger number (N > 30) for test statistic.
            rounds=rounds,
            kwargs={},
        )
        assert count > 0, f"TF record count: {count} must be greater than 0"
