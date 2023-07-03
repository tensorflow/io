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
"""ATDSDataset benchmark with dense tensors."""

import pytest
import tensorflow as tf

from tests.test_atds_avro.utils.data_source_registry import (
    TensorType,
    SMALL_NUM_RECORDS,
    SINGLE_PARTITION,
)
from tests.test_atds_avro.utils.atds_benchmark_utils import run_atds_benchmark


@pytest.mark.benchmark(
    group="dense_int32_0d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_int32_0d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        0,
        tf.int32,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_int32_1d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_int32_1d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        1,
        tf.int32,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_int32_2d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_int32_2d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        2,
        tf.int32,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_int64_0d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_int64_0d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        0,
        tf.int64,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_int64_1d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_int64_1d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        1,
        tf.int64,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_int64_2d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_int64_2d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        2,
        tf.int64,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_float32_0d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_float32_0d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        0,
        tf.float32,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_float32_1d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_float32_1d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        1,
        tf.float32,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_float32_2d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_float32_2d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        2,
        tf.float32,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_float64_0d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_float64_0d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        0,
        tf.float64,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_float64_1d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_float64_1d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        1,
        tf.float64,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_float64_2d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_float64_2d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        2,
        tf.float64,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_string_0d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_string_0d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        0,
        tf.string,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_string_1d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_string_1d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        1,
        tf.string,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_string_2d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_string_2d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        2,
        tf.string,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_bool_0d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_bool_0d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        0,
        tf.bool,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_bool_1d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_bool_1d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        1,
        tf.bool,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="dense_bool_2d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_dense_bool_2d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.DENSE,
        2,
        tf.bool,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )
