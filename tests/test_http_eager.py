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


class HTTPFSTest(tf.test.TestCase):
    """TestCase class to test the HTTP file system plugins
    functionality.
    """

    def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
        self.local_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_http", "LICENSE-2.0.txt"
        )
        self.remote_filename = "https://www.apache.org/licenses/LICENSE-2.0.txt"
        super().__init__(methodName)

    def setUp(self):  # pylint: disable=invalid-name
        with open(self.local_path) as f:
            self.local_lines = list(f)
        self.local_content = "".join(self.local_lines)
        super().setUp()

    def test_read_remote_file(self):
        """Test case for reading the entire content of the http file"""

        remote_content = tf.io.read_file(self.remote_filename)

        assert remote_content == self.local_content

    def test_dataset_from_remote_filename(self):
        """Test case to prepare a tf.data Dataset from a remote http file"""

        dataset = tf.data.TextLineDataset(self.remote_filename)
        i = 0
        for d in dataset:
            assert d == self.local_lines[i].rstrip()
            i += 1
        assert i == len(self.local_lines)

    def test_gfile_read(self):
        """Test case to read chunks of content from the http file"""

        remote_gfile = tf.io.gfile.GFile(self.remote_filename)
        assert remote_gfile.size() == len(self.local_content)
        offset_start = list(range(0, len(self.local_content), 100))
        offset_stop = offset_start[1:] + [len(self.local_content)]
        for (start, stop) in zip(offset_start, offset_stop):
            assert remote_gfile.read(100) == self.local_content[start:stop]

    def test_gfile_seek(self):
        """Test case to seek an offset after reading the content from the http file"""

        remote_gfile = tf.io.gfile.GFile(self.remote_filename)
        assert remote_gfile.read() == self.local_content
        assert remote_gfile.read() == ""
        remote_gfile.seek(0)
        assert remote_gfile.read() == self.local_content

    def test_gfile_tell(self):
        """Test case to tell the current position in the http file"""

        remote_gfile = tf.io.gfile.GFile(self.remote_filename)
        assert remote_gfile.tell() == 0
        remote_gfile.read(100)
        assert remote_gfile.tell() == 100


if __name__ == "__main__":
    tf.test.main()
