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
"""Utility functions for ATDS benchmarks."""

import avro.schema
import glob
import json
import os
import tensorflow as tf

from tests.test_atds_avro.utils.data_source import DataSource
from tests.test_atds_avro.utils.data_source_registry import (
    SMALL_NUM_RECORDS,
    get_canonical_name,
    get_data_source_from_registry,
)
from tests.test_atds_avro.utils.generator.tensor_generator import (
    IntTensorGenerator,
    FloatTensorGenerator,
    WordTensorGenerator,
    BoolTensorGenerator,
)
from tests.test_atds_avro.utils.generator.sparse_tensor_generator import (
    IntSparseTensorGenerator,
    FloatSparseTensorGenerator,
    WordSparseTensorGenerator,
    BoolSparseTensorGenerator,
    get_common_value_dist,
)
from tests.test_atds_avro.utils.generator.varlen_tensor_generator import (
    IntVarLenTensorGenerator,
    FloatVarLenTensorGenerator,
    WordVarLenTensorGenerator,
    BoolVarLenTensorGenerator,
)


from tensorflow_io.python.experimental.atds.dataset import ATDSDataset
from tests.test_atds_avro.utils.atds_writer import ATDSWriter
from tests.test_atds_avro.utils.benchmark_utils import benchmark_func


_AVRO_TO_DTYPE = {
    "int": tf.int32,
    "long": tf.int64,
    "float": tf.float32,
    "double": tf.float64,
    "boolean": tf.bool,
    "string": tf.string,
}

_AVRO_TO_DENSE_TENSOR_GENERATOR = {
    "int": IntTensorGenerator,
    "long": IntTensorGenerator,
    "float": FloatTensorGenerator,
    "double": FloatTensorGenerator,
    "boolean": BoolTensorGenerator,
    "string": WordTensorGenerator,
}

_AVRO_TO_SPARSE_TENSOR_GENERATOR = {
    "int": IntSparseTensorGenerator,
    "long": IntSparseTensorGenerator,
    "float": FloatSparseTensorGenerator,
    "double": FloatSparseTensorGenerator,
    "boolean": BoolSparseTensorGenerator,
    "string": WordSparseTensorGenerator,
}

_AVRO_TO_VARLEN_TENSOR_GENERATOR = {
    "int": IntVarLenTensorGenerator,
    "long": IntVarLenTensorGenerator,
    "float": FloatVarLenTensorGenerator,
    "double": FloatVarLenTensorGenerator,
    "boolean": BoolVarLenTensorGenerator,
    "string": WordVarLenTensorGenerator,
}


def get_features_from_data_source(writer, data_source):
    """Generates a dict of features from data source object

    Args:
      writer: ATDSWriter object
      data_source: DataSource object
    """
    scenario = data_source.scenario
    features = {
        feature_name: writer._get_atds_feature(scenario[feature_name])
        for feature_name in scenario
    }
    return features


def get_dataset(
    files,
    features,
    batch_size=1,
    shuffle_buffer_size=0,
    parallelism=os.cpu_count(),
    interleave_parallelism=0,
):
    """Generates a tf.data.Dataset from a datasource

    Args:
      files: A list of files
      features: Dict of features
      batch_size: (Optional.) Batch size for ATDS dataset
      shuffle_buffer_size: (Optional.) Size of the buffer used for shuffling. See
          tensorflow_io/python/experimental/atds/dataset.py for details.
          If unspecified, data is not shuffled.
      parallelism: (Optional.) Number of threads to use while decoding. Defaults
          to all available cores.
    """
    if interleave_parallelism == 0:
        dataset = ATDSDataset(
            filenames=files,
            batch_size=batch_size,
            features=features,
            shuffle_buffer_size=shuffle_buffer_size,
            num_parallel_calls=parallelism,
        )
    else:
        dataset = tf.data.Dataset.list_files(files)
        dataset = dataset.interleave(
            lambda filename: ATDSDataset(
                filenames=filename,
                batch_size=batch_size,
                features=features,
                shuffle_buffer_size=shuffle_buffer_size,
                num_parallel_calls=parallelism,
            ),
            cycle_length=interleave_parallelism,
            num_parallel_calls=interleave_parallelism,
        )
    return dataset.prefetch(1)


def _is_fully_defined_shape(shape):
    return -1 not in shape


def run_atds_benchmark(
    tensor_type, rank, dtype, num_records, partitions, batch_size, benchmark
):
    data_source_name = get_canonical_name(
        tensor_type, rank, dtype, num_records, partitions
    )
    data_source = get_data_source_from_registry(data_source_name)
    run_atds_benchmark_from_data_source(data_source, batch_size, benchmark)


def run_atds_benchmark_from_data_source(
    data_source,
    batch_size,
    benchmark,
    parallelism=tf.data.AUTOTUNE,
    interleave_parallelism=0,
    codec="null",
    shuffle_buffer_size=0,
    rounds=30,
):
    with ATDSWriter(codec=codec) as writer:
        dir_path = writer.write(data_source)
        pattern = os.path.join(dir_path, f"*.{writer.extension}")

        dataset = get_dataset(
            glob.glob(pattern),
            get_features_from_data_source(writer, data_source),
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            parallelism=parallelism,
            interleave_parallelism=interleave_parallelism,
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
        assert count > 0, f"ATDS record count: {count} must be greater than 0"
