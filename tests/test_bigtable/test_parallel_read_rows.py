# Copyright 2021 Google LLC
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

# disable module docstring for tests
# pylint: disable=C0114
# disable class docstring for tests
# pylint: disable=C0115

import os
from re import escape
from .bigtable_emulator import BigtableEmulator
from tensorflow_io.python.ops import core_ops
from tensorflow_io.python.ops.bigtable.bigtable_dataset_ops import BigtableClient
import tensorflow_io.python.ops.bigtable.bigtable_row_range as row_range
import tensorflow_io.python.ops.bigtable.bigtable_row_set as row_set
import tensorflow as tf
from tensorflow import test
import pytest
import sys


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
class BigtableParallelReadTest(test.TestCase):
    def setUp(self):
        self.emulator = BigtableEmulator()

    def tearDown(self):
        self.emulator.stop()

    def test_parallel_read(self):
        os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
        self.emulator.create_table(
            "fake_project",
            "fake_instance",
            "test-table",
            ["fam1", "fam2"],
            splits=["row005", "row010", "row015"],
        )

        values = [[f"[{i,j}]" for j in range(2)] for i in range(20)]
        flat_values = [value for row in values for value in row]

        ten = tf.constant(values)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        self.emulator.write_tensor(
            "fake_project",
            "fake_instance",
            "test-table",
            ten,
            ["row" + str(i).rjust(3, "0") for i in range(20)],
            ["fam1:col1", "fam2:col2"],
        )

        for r in table.parallel_read_rows(
            ["fam1:col1", "fam2:col2"],
            row_set=row_set.from_rows_or_ranges(row_range.infinite()),
        ):
            for c in r:
                self.assertTrue(c.numpy().decode() in flat_values)

    def test_not_parallel_read(self):
        os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
        self.emulator.create_table(
            "fake_project",
            "fake_instance",
            "test-table",
            ["fam1", "fam2"],
            splits=["row005", "row010", "row015"],
        )

        values = [[f"[{i,j}]" for j in range(2)] for i in range(20)]

        ten = tf.constant(values)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        self.emulator.write_tensor(
            "fake_project",
            "fake_instance",
            "test-table",
            ten,
            ["row" + str(i).rjust(3, "0") for i in range(20)],
            ["fam1:col1", "fam2:col2"],
        )

        dataset = table.parallel_read_rows(
            ["fam1:col1", "fam2:col2"],
            row_set=row_set.from_rows_or_ranges(row_range.infinite()),
            num_parallel_calls=2,
        )
        results = [[v.numpy().decode() for v in row] for row in dataset]
        self.assertEqual(repr(sorted(values)), repr(sorted(results)))

    def test_split_row_set(self):
        os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
        self.emulator.create_table(
            "fake_project",
            "fake_instance",
            "test-table",
            ["fam1", "fam2"],
            splits=["row005", "row010", "row015", "row020", "row025", "row030"],
        )

        values = [[f"[{i,j}]" for j in range(2)] for i in range(40)]

        ten = tf.constant(values)

        client = BigtableClient("fake_project", "fake_instance")

        self.emulator.write_tensor(
            "fake_project",
            "fake_instance",
            "test-table",
            ten,
            ["row" + str(i).rjust(3, "0") for i in range(40)],
            ["fam1:col1", "fam2:col2"],
        )

        rs = row_set.from_rows_or_ranges(row_range.infinite())

        num_parallel_calls = 2
        samples = [
            s
            for s in core_ops.bigtable_split_row_set_evenly(
                client._client_resource,
                rs._impl,
                "test-table",
                num_parallel_calls,
            )
        ]
        self.assertEqual(len(samples), num_parallel_calls)

        num_parallel_calls = 6
        samples = [
            s
            for s in core_ops.bigtable_split_row_set_evenly(
                client._client_resource,
                rs._impl,
                "test-table",
                num_parallel_calls,
            )
        ]

        # The emulator may return different samples each time, so we can't
        # expect an exact number, but it must be no more than num_parallel_calls
        self.assertLessEqual(len(samples), num_parallel_calls)

        num_parallel_calls = 1
        samples = [
            s
            for s in core_ops.bigtable_split_row_set_evenly(
                client._client_resource,
                rs._impl,
                "test-table",
                num_parallel_calls,
            )
        ]
        self.assertEqual(len(samples), num_parallel_calls)

    def test_split_empty(self):
        os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
        self.emulator.create_table(
            "fake_project",
            "fake_instance",
            "test-table",
            ["fam1", "fam2"],
        )

        client = BigtableClient("fake_project", "fake_instance")

        rs = row_set.from_rows_or_ranges(row_range.empty())

        num_parallel_calls = 2

        self.assertRaises(
            tf.errors.FailedPreconditionError,
            lambda: core_ops.bigtable_split_row_set_evenly(
                client._client_resource,
                rs._impl,
                "test-table",
                num_parallel_calls,
            ),
        )
