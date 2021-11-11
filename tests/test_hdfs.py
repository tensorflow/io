# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for HDFS file system"""

import os
import sys
import socket
import time
import tempfile
import tensorflow as tf
import tensorflow_io as tfio
import pytest


@pytest.mark.skipif(
    sys.platform in ("win32", "darwin"),
    reason="TODO HDFS not setup properly on macOS/Windows yet",
)
def test_read_file():
    """Test case for reading HDFS"""

    address = socket.gethostbyname(socket.gethostname())
    print(f"ADDRESS: {address}")

    body = b"1234567"
    tf.io.write_file(f"hdfs://{address}:9000/file.txt", body)

    content = tf.io.read_file(f"hdfs://{address}:9000/file.txt")
    print(f"CONTENT: {content}")
    assert content == body


@pytest.mark.skipif(
    sys.platform in ("win32", "darwin"),
    reason="TODO HDFS not setup properly on macOS/Windows yet",
)
def test_append_non_existing_file():
    """Test case for append a non-existing HDFS file"""

    address = socket.gethostbyname(socket.gethostname())
    print(f"ADDRESS: {address}")

    body = b"1234567"
    filepath = f"hdfs://{address}:9000/non-existing.txt"
    f = tf.io.gfile.GFile(filepath, "a")
    f.write(body)
    f.flush()

    content = tf.io.read_file(filepath)
    print(f"CONTENT: {content}")
    assert content == body
    f.close()


@pytest.mark.skipif(
    sys.platform in ("win32", "darwin", "linux"),
    reason="TODO HDFS not setup properly on macOS/Windows yet. For linux, current HDFS setup will throw:"
    "java.io.IOException: Failed to replace a bad datanode on the existing pipeline due to no more"
    "good datanodes being available to try. ",
)
def test_append_existing_file():
    """Test case for append an existing HDFS file"""

    address = socket.gethostbyname(socket.gethostname())
    print(f"ADDRESS: {address}")
    body1 = b"1234567"
    body2 = b"7654321"

    # create a new file
    filepath = f"hdfs://{address}:9000/to_be_appended.txt"
    tf.io.write_file(filepath, body1)

    # append to the file
    f = tf.io.gfile.GFile(filepath, "a")
    f.write(body2)
    f.flush()

    content = tf.io.read_file(filepath)
    print(f"CONTENT: {content}")
    assert content == body1 + body2
    f.close()
