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
"""Tests for hash_util module"""

import pytest

from tensorflow_io.python.experimental.benchmark.hash_util import \
  int_to_bytes


@pytest.mark.parametrize(
  ["value", "expected"], [
    (1000, b'\xe8\x03'),
    (-123, b'\x85'),
    (128, b'\x80\x00'),
    (0, b'\x00')
  ]
)
def test_int_to_bytes(value, expected):
  actual = int_to_bytes(value)
  assert actual == expected
