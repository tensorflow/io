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
import sys
import pytest

import tensorflow as tf
import tensorflow_io as tfio  # pylint: disable=unused-import

if sys.platform == "darwin":
    pytest.skip("TODO: http is failing on macOS with xdist", allow_module_level=True)


@pytest.fixture(scope="module")
def local_lines():
    local_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_http", "LICENSE-2.0.txt"
    )
    with open(local_path) as f:
        return list(f)


@pytest.fixture(scope="module")
def local_content():
    local_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_http", "LICENSE-2.0.txt"
    )
    with open(local_path) as f:
        local_lines = list(f)
    return "".join(local_lines)


@pytest.fixture(scope="module")
def remote_filename():
    return "https://www.apache.org/licenses/LICENSE-2.0.txt"


def test_read_remote_file(local_content, remote_filename):
    """Test case for reading the entire content of the http file"""

    remote_content = tf.io.read_file(remote_filename)

    assert remote_content == local_content


def test_dataset_from_remote_filename(local_lines, local_content, remote_filename):
    """Test case to prepare a tf.data Dataset from a remote http file"""

    dataset = tf.data.TextLineDataset(remote_filename)
    i = 0
    for d in dataset:
        assert d == local_lines[i].rstrip()
        i += 1
    assert i == len(local_lines)


def test_gfile_read(local_content, remote_filename):
    """Test case to read chunks of content from the http file"""

    remote_gfile = tf.io.gfile.GFile(remote_filename)
    assert remote_gfile.size() == len(local_content)
    offset_start = list(range(0, len(local_content), 100))
    offset_stop = offset_start[1:] + [len(local_content)]
    for (start, stop) in zip(offset_start, offset_stop):
        assert remote_gfile.read(100) == local_content[start:stop]


def test_gfile_seek(local_content, remote_filename):
    """Test case to seek an offset after reading the content from the http file"""

    remote_gfile = tf.io.gfile.GFile(remote_filename)
    assert remote_gfile.read() == local_content
    assert remote_gfile.read() == ""
    remote_gfile.seek(0)
    assert remote_gfile.read() == local_content


def test_gfile_tell(local_content, remote_filename):
    """Test case to tell the current position in the http file"""

    remote_gfile = tf.io.gfile.GFile(remote_filename)
    assert remote_gfile.tell() == 0
    remote_gfile.read(100)
    assert remote_gfile.tell() == 100


if __name__ == "__main__":
    tf.test.main()
