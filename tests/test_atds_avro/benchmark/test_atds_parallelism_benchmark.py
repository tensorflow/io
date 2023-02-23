# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""ATDS benchmark with parallelism."""

import pytest
import tensorflow as tf

from tensorflow_io.python.experimental.benchmark.data_source import \
  DataSource
from tensorflow_io.python.experimental.benchmark.data_source_registry import \
  LARGE_NUM_RECORDS
from tests.test_atds_avro.utils.atds_benchmark_utils import \
  run_atds_benchmark_from_data_source
from tests.test_benchmark.benchmark.utils.benchmark_utils import \
  MIXED_TYPES_SCENARIO


@pytest.mark.benchmark(group="parallelism",)
@pytest.mark.parametrize(
  ["batch_size", "shuffle_buffer_size", "codec", "parallelism"], [
    (128, 1024, "null", 1),
    (128, 1024, "null", 2),
    (128, 1024, "null", 3),
    (128, 1024, "null", 4),
    (128, 1024, "null", 5),
    (128, 1024, "null", 6),
    (128, 1024, "deflate", 1),
    (128, 1024, "deflate", 2),
    (128, 1024, "deflate", 3),
    (128, 1024, "deflate", 4),
    (128, 1024, "deflate", 5),
    (128, 1024, "deflate", 6),
    (128, 1024, "snappy", 1),
    (128, 1024, "snappy", 2),
    (128, 1024, "snappy", 3),
    (128, 1024, "snappy", 4),
    (128, 1024, "snappy", 5),
    (128, 1024, "snappy", 6),
  ]
)
def test_parallelism(batch_size, shuffle_buffer_size, codec, parallelism, benchmark):
  data_source = DataSource(
    scenario=MIXED_TYPES_SCENARIO,
    num_records=LARGE_NUM_RECORDS
  )
  run_atds_benchmark_from_data_source(data_source, batch_size, benchmark, parallelism=parallelism, codec=codec,
      shuffle_buffer_size=shuffle_buffer_size, rounds=10)

@pytest.mark.benchmark(group="parallelism",)
@pytest.mark.parametrize(
  ["batch_size", "shuffle_buffer_size", "parallelism", "interleave"], [
    (32, 1024, 1, 6),
    (32, 1024, 2, 3),
    (32, 1024, 3, 2),
    (32, 1024, 6, 1),
    (32, 1024, tf.data.AUTOTUNE, 1),
    (32, 1024, tf.data.AUTOTUNE, 2),
    (32, 1024, tf.data.AUTOTUNE, 3),
    (32, 1024, tf.data.AUTOTUNE, 6),
    (128, 1024, 1, 6),
    (128, 1024, 2, 3),
    (128, 1024, 3, 2),
    (128, 1024, 6, 1),
    (128, 1024, tf.data.AUTOTUNE, 1),
    (128, 1024, tf.data.AUTOTUNE, 2),
    (128, 1024, tf.data.AUTOTUNE, 3),
    (128, 1024, tf.data.AUTOTUNE, 6),
  ]
)
def test_parallelism_with_interleave(batch_size, shuffle_buffer_size, parallelism, interleave, benchmark):
  data_source = DataSource(
    scenario=MIXED_TYPES_SCENARIO,
    num_records=LARGE_NUM_RECORDS,
    partitions=6
  )
  run_atds_benchmark_from_data_source(data_source, batch_size, benchmark, parallelism=parallelism, interleave_parallelism=interleave,
      codec="deflate", shuffle_buffer_size=shuffle_buffer_size)
