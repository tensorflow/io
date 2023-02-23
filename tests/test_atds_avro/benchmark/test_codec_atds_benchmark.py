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
"""ATDS benchmark with different codecs."""

import pytest

from tensorflow_io.python.experimental.benchmark.data_source import \
  DataSource
from tensorflow_io.python.experimental.benchmark.data_source_registry import \
  SMALL_NUM_RECORDS
from tests.test_atds_avro.utils.atds_benchmark_utils import \
  run_atds_benchmark_from_data_source
from tests.test_benchmark.benchmark.utils.benchmark_utils import \
  MIXED_TYPES_SCENARIO


@pytest.mark.benchmark(group="codec",)
@pytest.mark.parametrize(
  ["batch_size", "codec"], [
    (128, "null"),
    (128, "deflate"),
    (128, "snappy")
  ]
)
def test_codec(batch_size, codec, benchmark):
  data_source = DataSource(
    scenario=MIXED_TYPES_SCENARIO,
    num_records=SMALL_NUM_RECORDS
  )
  run_atds_benchmark_from_data_source(data_source, batch_size, benchmark, codec=codec)
