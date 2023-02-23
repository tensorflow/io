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
"""ATDSDataset benchmark with all types of tensors for memory leak check."""

import pytest
import os

from tensorflow_io.python.experimental.benchmark.data_source_registry import \
  ALL_TYPES_DATA_SOURCE_NAME, get_data_source_from_registry

from tests.test_atds_avro.utils.atds_benchmark_utils import \
  run_atds_benchmark_from_data_source


@pytest.mark.skipif(
  os.getenv('ATDS_MEM_LEAK_CHECK') != "1",
  reason="This benchmark test is only used in memory leak check.")
@pytest.mark.benchmark(group="all_types_of_tensors",)
@pytest.mark.parametrize("batch_size", [(16)])
def test_all_types_of_tensors_for_memory_leak_check(batch_size, benchmark):
  data_source = get_data_source_from_registry(ALL_TYPES_DATA_SOURCE_NAME)
  shuffle_buffer_size = batch_size * 8
  run_atds_benchmark_from_data_source(
    data_source, batch_size, benchmark, codec="deflate",
    shuffle_buffer_size=shuffle_buffer_size, rounds=1)
