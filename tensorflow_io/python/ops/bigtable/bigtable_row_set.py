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

"""Module implementing basic functions for obtaining BigTable RowSets"""

from tensorflow_io.python.ops import core_ops
from tensorflow_io.python.ops.bigtable import bigtable_row_range
from typing import Union


class RowSet:
    """Object representing a set of rows by keeping a list of RowKeys and
    RowRanges that the set consists of."""

    def __init__(self, impl):
        self._impl = impl

    def __repr__(self) -> str:
        return core_ops.bigtable_print_row_set(self._impl).numpy()[0].decode()

    def append(self, row_or_range):
        if isinstance(row_or_range, str):
            core_ops.bigtable_row_set_append_row(self._impl, row_or_range)
        else:
            core_ops.bigtable_row_set_append_row_range(self._impl, row_or_range._impl)


def empty() -> RowSet:
    """Create an empty row set."""
    return RowSet(core_ops.bigtable_empty_row_set())


def from_rows_or_ranges(*args: Union[str, bigtable_row_range.RowRange]) -> RowSet:
    """Create a set from a row range.

    Args:
      *args: A row range (RowRange) which will be
          appended to an empty row set.
    Returns:
      RowSet: a set of rows containing the given row range.
    """
    row_set = empty()
    for row_or_range in args:
        row_set.append(row_or_range)

    return row_set


def intersect(row_set: RowSet, row_range: bigtable_row_range.RowRange) -> RowSet:
    """Modify a RowSet by intersecting its contents with a RowRange.

    All rows intersecting with the given range will be removed from the set
    and all row ranges will either be adjusted so that they do not cover
    anything beyond the given range or removed entirely (if they have an
    empty intersection with the given range).

    Args:
      row_set: A set (RowSet) which will be intersected.
      row_range (RowRange): The range with which this row set will be
          intersected.
    Returns:
      RowSet: an intersection of the given row set and row range.
    """
    return RowSet(core_ops.bigtable_row_set_intersect(row_set._impl, row_range._impl))
