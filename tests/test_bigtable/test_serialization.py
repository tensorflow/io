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
from .bigtable_emulator import BigtableEmulator
from tensorflow_io.python.ops.bigtable.bigtable_dataset_ops import BigtableClient
import tensorflow_io.python.ops.bigtable.bigtable_row_range as row_range
import tensorflow_io.python.ops.bigtable.bigtable_row_set as row_set
import tensorflow as tf
from tensorflow import test
from google.auth.credentials import AnonymousCredentials
from google.cloud.bigtable import Client
import datetime
import pytest
import sys


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
class BigtableReadTest(test.TestCase):
    def check_values(self, values, table, type_name, tf_dtype):
        for i, r in enumerate(
            table.read_rows(
                ["fam1:" + type_name],
                row_set=row_set.from_rows_or_ranges(row_range.infinite()),
                output_type=tf_dtype,
            )
        ):
            if tf_dtype in [tf.float64, tf.float32]:
                self.assertAlmostEqual(values[i].numpy(), r.numpy()[0])
            else:
                self.assertEqual(values[i].numpy(), r.numpy()[0])

    def setUp(self):
        self.emulator = BigtableEmulator()
        self.data = {
            "values": [i * 10 / 7 for i in range(10)],
            "float": [
                b"\x00\x00\x00\x00",
                b"?\xb6\xdbn",
                b"@6\xdbn",
                b"@\x89$\x92",
                b"@\xb6\xdbn",
                b"@\xe4\x92I",
                b"A\t$\x92",
                b"A \x00\x00",
                b"A6\xdbn",
                b"AM\xb6\xdb",
            ],
            "double": [
                b"\x00\x00\x00\x00\x00\x00\x00\x00",
                b"?\xf6\xdbm\xb6\xdbm\xb7",
                b"@\x06\xdbm\xb6\xdbm\xb7",
                b"@\x11$\x92I$\x92I",
                b"@\x16\xdbm\xb6\xdbm\xb7",
                b"@\x1c\x92I$\x92I%",
                b"@!$\x92I$\x92I",
                b"@$\x00\x00\x00\x00\x00\x00",
                b"@&\xdbm\xb6\xdbm\xb7",
                b"@)\xb6\xdbm\xb6\xdbn",
            ],
            "int32": [
                b"\x00\x00\x00\x00",
                b"\x00\x00\x00\x01",
                b"\x00\x00\x00\x02",
                b"\x00\x00\x00\x04",
                b"\x00\x00\x00\x05",
                b"\x00\x00\x00\x07",
                b"\x00\x00\x00\x08",
                b"\x00\x00\x00\n",
                b"\x00\x00\x00\x0b",
                b"\x00\x00\x00\x0c",
            ],
            "int64": [
                b"\x00\x00\x00\x00\x00\x00\x00\x00",
                b"\x00\x00\x00\x00\x00\x00\x00\x01",
                b"\x00\x00\x00\x00\x00\x00\x00\x02",
                b"\x00\x00\x00\x00\x00\x00\x00\x04",
                b"\x00\x00\x00\x00\x00\x00\x00\x05",
                b"\x00\x00\x00\x00\x00\x00\x00\x07",
                b"\x00\x00\x00\x00\x00\x00\x00\x08",
                b"\x00\x00\x00\x00\x00\x00\x00\n",
                b"\x00\x00\x00\x00\x00\x00\x00\x0b",
                b"\x00\x00\x00\x00\x00\x00\x00\x0c",
            ],
            "bool": [
                b"\x00",
                b"\xff",
                b"\xff",
                b"\xff",
                b"\xff",
                b"\xff",
                b"\xff",
                b"\xff",
                b"\xff",
                b"\xff",
            ],
        }

        os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
        self.emulator.create_table(
            "fake_project", "fake_instance", "test-table", ["fam1"]
        )

        client = Client(
            project="fake_project", credentials=AnonymousCredentials(), admin=True
        )
        table = client.instance("fake_instance").table("test-table")

        for type_name in ["float", "double", "int32", "int64", "bool"]:
            rows = []
            for i, value in enumerate(self.data[type_name]):
                row_key = "row" + str(i).rjust(3, "0")
                row = table.direct_row(row_key)
                row.set_cell(
                    "fam1", type_name, value, timestamp=datetime.datetime.utcnow()
                )
                rows.append(row)
            table.mutate_rows(rows)

    def tearDown(self):
        self.emulator.stop()

    def test_float_xdr(self):
        values = tf.constant(self.data["values"], dtype=tf.float32)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        self.check_values(values, table, "float", tf.float32)

    def test_double_xdr(self):
        values = tf.constant(self.data["values"], dtype=tf.float64)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        self.check_values(values, table, "double", tf.float64)

    def test_int64_xdr(self):
        values = tf.cast(tf.constant(self.data["values"]), dtype=tf.int64)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        self.check_values(values, table, "int64", tf.int64)

    def test_int32_xdr(self):
        values = tf.cast(tf.constant(self.data["values"]), dtype=tf.int32)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        self.check_values(values, table, "int32", tf.int32)

    def test_bool_xdr(self):
        values = tf.cast(tf.constant(self.data["values"]), dtype=tf.bool)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        self.check_values(values, table, "bool", tf.bool)
