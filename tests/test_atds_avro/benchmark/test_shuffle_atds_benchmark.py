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
"""ATDSDataset shuffle benchmark."""

import glob
import os
import pytest

from tensorflow_io.python.experimental.benchmark.data_source import \
  DataSource
from tensorflow_io.python.experimental.benchmark.data_source_registry import \
  SMALL_NUM_RECORDS
from tensorflow_io.python.experimental.atds.atds_writer import \
  ATDSWriter
from tests.test_atds_avro.utils.atds_benchmark_utils import \
  get_dataset, get_features_from_data_source, run_atds_benchmark_from_data_source
from tests.test_benchmark.benchmark.utils.benchmark_utils import \
  MIXED_TYPES_SCENARIO, benchmark_func


@pytest.mark.benchmark(group="shuffle",)
@pytest.mark.parametrize(
  ["batch_size", "shuffle_buffer_size"], [
    (128, 0),
    (128, 64), # shuffle_buffer_size < batch_size (imperfect shuffle)
    (128, 512), # shuffle_buffer_size > batch_size (imperfect shuffle)
    (128, 1024), # shuffle_buffer_size + batch_size > num_records (perfect shuffle)
  ]
)
def test_in_ops_shuffle(batch_size, shuffle_buffer_size, benchmark):
  data_source = DataSource(
    scenario=MIXED_TYPES_SCENARIO,
    num_records=SMALL_NUM_RECORDS
  )
  run_atds_benchmark_from_data_source(data_source, batch_size, benchmark, shuffle_buffer_size=shuffle_buffer_size)

@pytest.mark.benchmark(group="shuffle",)
@pytest.mark.parametrize(
  ["batch_size", "shuffle_buffer_size"], [
    (128, 64), # shuffle_buffer_size < batch_size (imperfect shuffle)
    (128, 512), # shuffle_buffer_size > batch_size (imperfect shuffle)
    (128, 1024), # shuffle_buffer_size + batch_size > num_records (perfect shuffle)
  ]
)
def test_unbatch_shuffle_batch(batch_size, shuffle_buffer_size, benchmark):
  data_source = DataSource(
    scenario=MIXED_TYPES_SCENARIO,
    num_records=SMALL_NUM_RECORDS
  )
  with ATDSWriter(codec="null") as writer:
    dir_path = writer.write(data_source)
    pattern = os.path.join(dir_path, f"*.{writer.extension}")

    dataset = get_dataset(glob.glob(pattern), get_features_from_data_source(writer, data_source),
        batch_size=batch_size)
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    count = benchmark.pedantic(
        target=benchmark_func,
        args=[dataset],
        iterations=2,
        rounds=100,
        kwargs={}
    )
    assert count > 0, f"ATDS record count: {count} must be greater than 0"
