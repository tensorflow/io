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
"""Utility functions for hashing"""


def int_to_bytes(x):
    # Add one bit because a signed N-bit int can only represent up to 2^(N-1) - 1
    # (instead of an unsigned N-bit int which can represent up to 2^N - 1).
    # For example, 128 requires 9 bits (therefore two bytes) in twos complement.
    return x.to_bytes(x.bit_length() // 8 + 1, byteorder="little", signed=True)
