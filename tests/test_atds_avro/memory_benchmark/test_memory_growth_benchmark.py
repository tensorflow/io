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
"""ATDSDataset benchmark for memory growth test."""

import pytest
import os
import tensorflow as tf

from tensorflow_io.python.experimental.benchmark.data_source import \
  DataSource
from tensorflow_io.python.experimental.benchmark.generator.\
  tensor_generator import IntTensorGenerator

from tests.test_atds_avro.utils.atds_benchmark_utils import \
  run_atds_benchmark_from_data_source


@pytest.mark.benchmark(group="memory_growth",)
@pytest.mark.parametrize("n", [(1), (2), (4), (8), (32), (128), (512), (1024)])
def test_memory_growth(n, benchmark):
  batch_size = 128
  # n is the shuffle buffer size to batch size ratio.
  shuffle_buffer_size = batch_size * n
  scenario = {
    "tensor": IntTensorGenerator(tf.TensorSpec(shape=[16], dtype=tf.int32))
  }
  # A fixed number of records that covers all ratio n.
  num_records = 720 * 9 * 1024
  data_source = DataSource(scenario=scenario, num_records=num_records)
  run_atds_benchmark_from_data_source(
    data_source, batch_size, benchmark, codec="null",
    shuffle_buffer_size=shuffle_buffer_size, rounds=1)
