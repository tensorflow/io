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
"""Global DataSource registry with predefined DataSource used in benchmark"""

from enum import Enum

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import registry

from tests.test_atds_avro.utils.data_source import DataSource
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
    ValueDistribution,
)
from tests.test_atds_avro.utils.generator.varlen_tensor_generator import (
    DimensionDistribution,
    IntVarLenTensorGenerator,
    FloatVarLenTensorGenerator,
    WordVarLenTensorGenerator,
    BoolVarLenTensorGenerator,
    DIM_DISTRIBUTION_TO_RANGE,
)

SMALL_NUM_RECORDS = 1024
LARGE_NUM_RECORDS = 12 * 1024

SINGLE_PARTITION = 1
MULTIPLE_PARTITION = 6

FILE_PARALLELISM = 2

_data_source_registry = registry.Registry("data source")

# Data source name with all types of tensors.
ALL_TYPES_DATA_SOURCE_NAME = "all_types_data_source_name"


class TensorType(Enum):
    """Type of tensors used in benchmark"""

    DENSE = 1
    SPARSE = 2
    VARLEN = 3


def get_canonical_name(tensor_type, rank, dtype, num_records, partitions):
    """Get canonical name which is used as key in global data source registry.

    Args:
      tensor_type: A TensorType enum.
      rank: An int to represent the rank of tensor.
      dtype: tf.dtypes.DType.
      num_records: Number of records.
      partitions: Number of file partitions.

    Returns:
      The canonical name to represent such data source in registry.

    Raises:
      TypeError: If tensor_type is not TensorType.
      ValueError: if rank is unknown or negative.
    """
    if not isinstance(tensor_type, TensorType):
        raise TypeError(
            "Input tensor_type must be a TensorType enum" f" but found {tensor_type}"
        )

    if rank is None or rank < 0:
        raise ValueError("Input rank must not be None or negative. Found {rank}.")

    return f"{tensor_type.name}_{rank}D_{dtype.name}_{num_records}_{partitions}"


def get_data_source_registry():
    """Get the global data source registry. If the registry is empty,
    initialize the registry with predefined data sources."""
    global _data_source_registry
    if not _data_source_registry.list():
        _init_data_source_registry(_data_source_registry)
    return _data_source_registry


def _init_data_source_registry(registry):
    shapes = [[], [128], [64, 64]]
    dtypes = [tf.int32, tf.int64, tf.float32, tf.float64, tf.string, tf.bool]

    # Register data source with dense tensors.
    dense_generators = [
        IntTensorGenerator,
        IntTensorGenerator,
        FloatTensorGenerator,
        FloatTensorGenerator,
        WordTensorGenerator,
        BoolTensorGenerator,
    ]
    for cls, dtype in zip(dense_generators, dtypes):
        for shape in shapes:
            name = get_canonical_name(
                TensorType.DENSE,
                rank=len(shape),
                dtype=dtype,
                num_records=SMALL_NUM_RECORDS,
                partitions=SINGLE_PARTITION,
            )
            generator = cls(tf.TensorSpec(shape=shape, dtype=dtype))
            registry.register(
                candidate=DataSource(
                    scenario={name: generator},
                    num_records=SMALL_NUM_RECORDS,
                    partitions=SINGLE_PARTITION,
                ),
                name=name,
            )

    # Register data source with sparse tensors.
    sparse_generators = [
        IntSparseTensorGenerator,
        IntSparseTensorGenerator,
        FloatSparseTensorGenerator,
        FloatSparseTensorGenerator,
        WordSparseTensorGenerator,
        BoolSparseTensorGenerator,
    ]
    for cls, dtype in zip(sparse_generators, dtypes):
        for shape in shapes:
            if len(shape) == 0:
                # Skip scalars for sparse tensors
                continue
            name = get_canonical_name(
                TensorType.SPARSE,
                rank=len(shape),
                dtype=dtype,
                num_records=SMALL_NUM_RECORDS,
                partitions=SINGLE_PARTITION,
            )
            value_dist = ValueDistribution.SMALL_NUM_VALUE  # 5 to 10 elements
            generator = cls(tf.SparseTensorSpec(shape=shape, dtype=dtype), value_dist)
            registry.register(
                candidate=DataSource(
                    scenario={name: generator},
                    num_records=SMALL_NUM_RECORDS,
                    partitions=SINGLE_PARTITION,
                ),
                name=name,
            )

    # Register data source with varlen tensors.
    varlen_generators = [
        IntVarLenTensorGenerator,
        IntVarLenTensorGenerator,
        FloatVarLenTensorGenerator,
        FloatVarLenTensorGenerator,
        WordVarLenTensorGenerator,
        BoolVarLenTensorGenerator,
    ]
    varlen_shapes = [[None]]
    for cls, dtype in zip(varlen_generators, dtypes):
        for shape in varlen_shapes:
            rank = len(shape)
            name = get_canonical_name(
                TensorType.VARLEN,
                rank=rank,
                dtype=dtype,
                num_records=SMALL_NUM_RECORDS,
                partitions=SINGLE_PARTITION,
            )
            dim_dist = DimensionDistribution.LARGE_DIM  # dim is between 5 to 10

            generator = cls(tf.SparseTensorSpec(shape=shape, dtype=dtype), dim_dist)
            registry.register(
                candidate=DataSource(
                    scenario={name: generator},
                    num_records=SMALL_NUM_RECORDS,
                    partitions=SINGLE_PARTITION,
                ),
                name=name,
            )

    # Registry data source with all types of tensors
    scenario = {}
    for key in registry.list():
        data_source = registry.lookup(key)
        scenario = {**scenario, **data_source.scenario}

    # small num records for memory leak check
    registry.register(
        candidate=DataSource(
            scenario=scenario,
            num_records=SMALL_NUM_RECORDS,
            partitions=MULTIPLE_PARTITION,
        ),
        name=ALL_TYPES_DATA_SOURCE_NAME,
    )


def get_data_source_from_registry(name):
    registry = get_data_source_registry()
    return registry.lookup(name)
