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
from . import pbt_C
from typing import Union


def empty() -> pbt_C.RowSet:
  """Create an empty row set."""
  return pbt_C.RowSet()


def from_rows_or_ranges(*args: Union[str, pbt_C.RowRange]) -> pbt_C.RowSet:
  """ Create a set from a row range.

  Args:
    *args: A row range (RowRange) which will be
        appended to an empty row set.
  Returns:
    RowSet: a set of rows containing the given row range.
  """
  row_set = pbt_C.RowSet()
  for row_or_range in args:
    row_set.append(row_or_range)

  return row_set


def intersect(row_set, row_range: pbt_C.RowRange) -> pbt_C.RowSet:
  """ Modify a row set by intersecting its contents with a row range.

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
  return row_set.intersect(row_range)
