# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for HTTP file system"""

import os

import tensorflow as tf
import tensorflow_io as tfio  # pylint: disable=unused-import


def test_http_file_system():
    """Test case for http file system"""
    local_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_http", "LICENSE-2.0.txt"
    )
    with open(local_path) as f:
        lines = list(f)
    local = "".join(lines)

    filename = "http://www.apache.org/licenses/LICENSE-2.0.txt"
    remote = tf.io.read_file(filename)

    assert remote == local

    dataset = tf.data.TextLineDataset(filename)
    i = 0
    for d in dataset:
        assert d == lines[i].rstrip()
        i += 1
    assert i == len(lines)

    remote = tf.io.gfile.GFile(filename)
    assert remote.size() == len(local)
    offset_start = list(range(0, len(local), 100))
    offset_stop = offset_start[1:] + [len(local)]
    for (start, stop) in zip(offset_start, offset_stop):
        assert remote.read(100) == local[start:stop]


if __name__ == "__main__":
    test.main()
