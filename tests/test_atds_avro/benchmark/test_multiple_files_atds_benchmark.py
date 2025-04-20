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
"""ATDS benchmark with multiple files."""

import pytest

from tests.test_atds_avro.utils.data_source import DataSource
from tests.test_atds_avro.utils.data_source_registry import (
    LARGE_NUM_RECORDS,
    MULTIPLE_PARTITION,
)
from tests.test_atds_avro.utils.atds_benchmark_utils import (
    run_atds_benchmark_from_data_source,
)
from tests.test_atds_avro.utils.benchmark_utils import MIXED_TYPES_SCENARIO


@pytest.mark.benchmark(
    group="multi_partition",
)
@pytest.mark.parametrize(["batch_size", "partitions"], [(128, MULTIPLE_PARTITION)])
def test_multiple_partitions(batch_size, partitions, benchmark):
    data_source = DataSource(
        scenario=MIXED_TYPES_SCENARIO,
        num_records=LARGE_NUM_RECORDS,
        partitions=partitions,
    )
    run_atds_benchmark_from_data_source(data_source, batch_size, benchmark)
