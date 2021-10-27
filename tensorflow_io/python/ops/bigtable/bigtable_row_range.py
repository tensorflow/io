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

def infinite() -> pbt_C.RowRange:
  """Create a infinite row range."""
  return pbt_C.infinite_row_range()


def starting_at(row_key: str) -> pbt_C.RowRange:
  """Create a row range staring at given row (inclusive).

  Args:
    row_key (str): The starting row key of the range (inclusive).
  Returns:
    RowRange: The row range which starts at `row_key` (inclusive).
  """
  return pbt_C.starting_at_row_range(row_key)


def ending_at(row_key: str) -> pbt_C.RowRange:
  """Create a row range ending at given row (inclusive).

  Args:
    row_key (str): The ending row key of the range (inclusive).
  Returns:
    RowRange: The row range which ends at `row_key` (inclusive).
  """
  return pbt_C.ending_at_row_range(row_key)


def empty() -> pbt_C.RowRange:
  """Create an empty row range."""
  return pbt_C.empty_row_range()


def prefix(prefix_str: str) -> pbt_C.RowRange:
  """Create a range of all rows starting with a given prefix.

  Args:
    prefix_str (str): The prefix with which all rows start
  Returns:
    RowRange: The row range of all rows starting with the given prefix.
  """
  return pbt_C.prefix_row_range(prefix_str)


def right_open(start: str, end: str) -> pbt_C.RowRange:
  """Create a row range exclusive at the start and inclusive at the end.

  Args:
    start (str): The start of the row range (inclusive).
    end (str): The end of the row range (exclusive).
  Returns:
    RowRange: The row range between the `start` and `end`.
  """
  return pbt_C.right_open_row_range(start, end)


def left_open(start: str, end: str) -> pbt_C.RowRange:
  """Create a row range inclusive at the start and exclusive at the end.

  Args:
    start (str): The start of the row range (exclusive).
    end (str): The end of the row range (inclusive).
  Returns:
    RowRange: The row range between the `start` and `end`.
  """
  return pbt_C.left_open_row_range(start, end)


def open_range(start: str, end: str) -> pbt_C.RowRange:
  """Create a row range exclusive at both the start and the end.

  Args:
    start (str): The start of the row range (exclusive).
    end (str): The end of the row range (exclusive).
  Returns:
    RowRange: The row range between the `start` and `end`.
  """
  return pbt_C.open_row_range(start, end)


def closed_range(start: str, end: str) -> pbt_C.RowRange:
  """Create a row range inclusive at both the start and the end.

  Args:
    start (str): The start of the row range (inclusive).
    end (str): The end of the row range (inclusive).
  Returns:
    RowRange: The row range between the `start` and `end`.
  """
  return pbt_C.closed_row_range(start, end)
