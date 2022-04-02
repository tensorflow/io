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
from tensorflow_io.python.ops import core_ops
import tensorflow_io.python.ops.bigtable.bigtable_row_range as row_range
import tensorflow_io.python.ops.bigtable.bigtable_row_set as row_set
from tensorflow import test


class RowRangeTest(test.TestCase):
    def test_infinite(self):
        self.assertEqual("", repr(row_range.infinite()))

    def test_starting_at(self):
        expected = 'start_key_closed: "row1"\n'
        self.assertEqual(expected, repr(row_range.starting_at("row1")))

    def test_ending_at(self):
        expected = 'start_key_closed: ""\n' 'end_key_closed: "row1"\n'
        self.assertEqual(expected, repr(row_range.ending_at("row1")))

    def test_empty(self):
        expected = 'start_key_open: ""\n' + 'end_key_open: "\\000"\n'

        self.assertEqual(expected, repr(row_range.empty()))

    def test_prefix(self):
        expected = 'start_key_closed: "row1"\n' 'end_key_open: "row2"\n'
        self.assertEqual(expected, repr(row_range.prefix("row1")))

    def test_right_open(self):
        expected = 'start_key_closed: "row1"\n' 'end_key_open: "row2"\n'
        self.assertEqual(expected, repr(row_range.right_open("row1", "row2")))

    def test_left_open(self):
        expected = 'start_key_open: "row1"\n' 'end_key_closed: "row2"\n'
        self.assertEqual(expected, repr(row_range.left_open("row1", "row2")))

    def test_open(self):
        expected = 'start_key_open: "row1"\n' 'end_key_open: "row2"\n'
        self.assertEqual(expected, repr(row_range.open_range("row1", "row2")))

    def test_closed(self):
        expected = 'start_key_closed: "row1"\n' + 'end_key_closed: "row2"\n'
        self.assertEqual(expected, repr(row_range.closed_range("row1", "row2")))


class TestRowSet(test.TestCase):
    def test_empty(self):
        expected = ""
        self.assertEqual(expected, repr(row_set.empty()))

    def test_append_row(self):
        r_set = row_set.empty()
        r_set.append("row1")
        expected = 'row_keys: "row1"\n'
        self.assertEqual(expected, repr(r_set))

    def test_append_row_range(self):
        r_set = row_set.empty()
        r_set.append(row_range.closed_range("row1", "row2"))
        expected = (
            "row_ranges {\n"
            '  start_key_closed: "row1"\n'
            '  end_key_closed: "row2"\n'
            "}\n"
        )
        self.assertEqual(expected, repr(r_set))

    def test_from_rows_or_ranges(self):
        expected = (
            'row_keys: "row3"\n'
            'row_keys: "row6"\n'
            "row_ranges {\n"
            '  start_key_closed: "row1"\n'
            '  end_key_closed: "row2"\n'
            "}\n"
            "row_ranges {\n"
            '  start_key_open: "row4"\n'
            '  end_key_open: "row5"\n'
            "}\n"
        )

        r_set = row_set.from_rows_or_ranges(
            row_range.closed_range("row1", "row2"),
            "row3",
            row_range.open_range("row4", "row5"),
            "row6",
        )
        self.assertEqual(expected, repr(r_set))

    def test_intersect(self):
        r_set = row_set.from_rows_or_ranges(row_range.open_range("row1", "row5"))
        r_set = row_set.intersect(r_set, row_range.closed_range("row3", "row7"))
        expected = (
            "row_ranges {\n" + '  start_key_closed: "row3"\n' + "  end_key_open: "
            '"row5"\n' + "}\n"
        )
        self.assertEqual(expected, repr(r_set))
