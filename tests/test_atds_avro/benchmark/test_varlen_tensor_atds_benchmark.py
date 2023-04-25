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
"""ATDSDataset benchmark with varlen tensors."""

import pytest
import tensorflow as tf

from tests.test_atds_avro.utils.data_source_registry import (
    TensorType,
    SMALL_NUM_RECORDS,
    SINGLE_PARTITION,
)
from tests.test_atds_avro.utils.atds_benchmark_utils import run_atds_benchmark


@pytest.mark.benchmark(
    group="varlen_int32_1d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_varlen_int32_1d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.VARLEN,
        1,
        tf.int32,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="varlen_int64_1d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_varlen_int64_1d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.VARLEN,
        1,
        tf.int64,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="varlen_float32_1d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_varlen_float32_1d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.VARLEN,
        1,
        tf.float32,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="varlen_float64_1d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_varlen_float64_1d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.VARLEN,
        1,
        tf.float64,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="varlen_string_1d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_varlen_string_1d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.VARLEN,
        1,
        tf.string,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )


@pytest.mark.benchmark(
    group="varlen_bool_1d",
)
@pytest.mark.parametrize("batch_size", [(128)])
def test_varlen_bool_1d(batch_size, benchmark):
    run_atds_benchmark(
        TensorType.VARLEN,
        1,
        tf.bool,
        SMALL_NUM_RECORDS,
        SINGLE_PARTITION,
        batch_size,
        benchmark,
    )
