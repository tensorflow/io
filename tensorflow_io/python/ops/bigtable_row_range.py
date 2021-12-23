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

"""Module implementing basic functions for obtaining BigTable RowRanges"""

from tensorflow_io.python.ops import core_ops


class RowRange:
    """Object representing a range of rows."""

    def __init__(self, impl) -> None:
        self._impl = impl

    def __repr__(self) -> str:
        print(self._impl)
        return core_ops.bigtable_print_row_range(self._impl).numpy()[0].decode()


def infinite() -> RowRange:
    """Create a infinite row range.
    Returns:
      RowRange: Infinite RowRange."""
    return RowRange(core_ops.bigtable_row_range("", False, "", False))


def starting_at(row_key: str) -> RowRange:
    """Create a row range staring at given row (inclusive).

    Args:
      row_key (str): The starting row key of the range (inclusive).
    Returns:
      RowRange: The row range which starts at `row_key` (inclusive).
    """
    return RowRange(core_ops.bigtable_row_range(row_key, False, "", False))


def ending_at(row_key: str) -> RowRange:
    """Create a row range ending at given row (inclusive).

    Args:
      row_key (str): The ending row key of the range (inclusive).
    Returns:
      RowRange: The row range which ends at `row_key` (inclusive).
    """
    return RowRange(core_ops.bigtable_row_range("", False, row_key, False))


def empty() -> RowRange:
    """Create an empty row range.
    Returns:
      RowRange: Empty RowRange."""
    return RowRange(core_ops.bigtable_empty_row_range())


def prefix(prefix_str: str) -> RowRange:
    """Create a range of all rows starting with a given prefix.

    Args:
      prefix_str (str): The prefix with which all rows start
    Returns:
      RowRange: The row range of all rows starting with the given prefix.
    """
    return RowRange(core_ops.bigtable_prefix_row_range(prefix_str))


def right_open(start: str, end: str) -> RowRange:
    """Create a row range exclusive at the start and inclusive at the end.

    Args:
      start (str): The start of the row range (inclusive).
      end (str): The end of the row range (exclusive).
    Returns:
      RowRange: The row range between the `start` and `end`.
    """
    return RowRange(core_ops.bigtable_row_range(start, False, end, True))


def left_open(start: str, end: str) -> RowRange:
    """Create a row range inclusive at the start and exclusive at the end.

    Args:
      start (str): The start of the row range (exclusive).
      end (str): The end of the row range (inclusive).
    Returns:
      RowRange: The row range between the `start` and `end`.
    """
    return RowRange(core_ops.bigtable_row_range(start, True, end, False))


def open_range(start: str, end: str) -> RowRange:
    """Create a row range exclusive at both the start and the end.

    Args:
      start (str): The start of the row range (exclusive).
      end (str): The end of the row range (exclusive).
    Returns:
      RowRange: The row range between the `start` and `end`.
    """
    return RowRange(core_ops.bigtable_row_range(start, True, end, True))


def closed_range(start: str, end: str) -> RowRange:
    """Create a row range inclusive at both the start and the end.

    Args:
      start (str): The start of the row range (inclusive).
      end (str): The end of the row range (inclusive).
    Returns:
      RowRange: The row range between the `start` and `end`.
    """
    return RowRange(core_ops.bigtable_row_range(start, False, end, False))
