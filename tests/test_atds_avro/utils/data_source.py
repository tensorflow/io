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
"""DataSource"""

import hashlib

from tests.test_atds_avro.utils.hash_util import int_to_bytes


class DataSource:
    """DataSource describes properties in a benchmark data.

    DataSource contains the metadata of a benchmark data including the total
    number of records, the number of partitioned files and a scenario. A scenario
    defines the features used in benchmark with the feature name, tensor spec,
    and Generator used to generate the value. DataSource can be consumed by
    FileWriter to generate the benchmark data descrbied by itself.
    """

    def __init__(self, scenario, num_records, partitions=1):
        """Create a new DataSource.

        Args:
          scenario: A dict with feature name as key and Generator as value.
            Scenario defines the features used in benchmark. Generator contains
            tensor spec and the distribution to generate the tensor value.
          num_records: An int defines total number of records in this data.
          partitions: An int defines the number of partitioned files in this data.
            Each partition can have different number of records. However, the total
            number of records must be num_records.

        Raises:
          ValueError: If num_records or partitions is negative or partitions is
            zero but num_records is greater than zero.
        """
        if num_records < 0:
            raise ValueError(
                "Number of records in DataSource must not be negative"
                f" but got {num_records}."
            )
        if partitions < 0:
            raise ValueError(
                "Partition number in DataSource must not be negative"
                f" but got {partitions}."
            )
        if partitions == 0 and num_records > 0:
            raise ValueError(
                "Cannot have zero partitions in DataSource with"
                f"non-zero num_records ({num_records})."
            )

        self._scenario = scenario
        self._num_records = num_records
        self._partitions = partitions

    @property
    def scenario(self):
        """Return the scenario of the benchmark data.

        The scenario is a dict with feature name as key and Generator as value.
        """
        return self._scenario

    @property
    def num_records(self):
        """Return the total number of records in this data as int."""
        return self._num_records

    @property
    def partitions(self):
        """Return the number of partitioned files in this data as int."""
        return self._partitions

    def hash_code(self):
        """Return the consistent hashed code of the DataSource in hex str.

        The hashed code can be used as the path for data source cache.

        Returns:
          A hex str generated by hashing algorithm.
        """
        m = hashlib.sha256()
        # Step 1: hash sorted scenario dict
        for name in sorted(self.scenario):
            generator = self.scenario[name]
            m.update(name.encode())
            m.update(generator.hash_code().encode())

        # Step 2: hash num_records and partitions
        m.update(int_to_bytes(self.num_records))
        m.update(int_to_bytes(self.partitions))

        # Step 3: return hashed str in hex.
        return m.hexdigest()
