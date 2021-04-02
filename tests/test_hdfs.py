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

# Use modular file system plugins from tfio instead of the legacy implementation
# from tensorflow.
os.environ["TF_USE_MODULAR_FILESYSTEM"] = "true"
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
    print("ADDRESS: {}".format(address))

    body = b"1234567"
    tf.io.write_file("hdfs://{}:9000/file.txt".format(address), body)

    content = tf.io.read_file("hdfs://{}:9000/file.txt".format(address))
    print("CONTENT: {}".format(content))
    assert content == body
