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
"""Tests for Azure File System."""

import os
import sys
import pytest

import tensorflow as tf
import tensorflow_io as tfio  # pylint: disable=unused-import

# Note: export TF_AZURE_USE_DEV_STORAGE=1 to enable emulation

if sys.platform == "darwin":
    pytest.skip("TODO: skip macOS", allow_module_level=True)


class AZFSTest(tf.test.TestCase):
    """[summary]

    Args:
      test {[type]} -- [description]
    """

    def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name

        os.environ["TF_AZURE_USE_DEV_STORAGE"] = "1"

        self.account = "devstoreaccount1"
        self.container = "aztest"
        self.path_root = "az://" + os.path.join(self.account, self.container)
        super().__init__(methodName)

    def _path_to(self, path):
        return os.path.join(self.path_root, path)

    def setUp(self):  # pylint: disable=invalid-name
        super().setUp()
        if not tf.io.gfile.isdir(self.path_root):
            tf.io.gfile.makedirs(self.path_root)

    def test_exists(self):
        self.assertTrue(tf.io.gfile.isdir(self.path_root))

    def test_also_works_with_full_dns_name(self):
        """Test the file system also works when we're given
        a path of the form
        az://<account>.blob.core.windows.net/<container>/<path>
        """
        file_name = self.account + ".blob.core.windows.net" + self.container
        if not tf.io.gfile.isdir(file_name):
            tf.io.gfile.makedirs(file_name)

        self.assertTrue(tf.io.gfile.isdir(file_name))

    def test_create_file(self):
        """Test create file."""
        # Setup and check preconditions.
        file_name = self._path_to("testfile")
        if tf.io.gfile.exists(file_name):
            tf.io.gfile.remove(file_name)
        # Create file.
        with tf.io.gfile.GFile(file_name, "w") as w:
            w.write("")
        # Check that file was created.
        self.assertTrue(tf.io.gfile.exists(file_name))

        tf.io.gfile.remove(file_name)

    def test_write_read_file(self):
        """Test write/read file."""
        # Setup and check preconditions.
        file_name = self._path_to("writereadfile")
        if tf.io.gfile.exists(file_name):
            tf.io.gfile.remove(file_name)

        # Write data.
        with tf.io.gfile.GFile(file_name, "w") as w:
            w.write("Hello\n, world!")

        # Read data.
        with tf.io.gfile.GFile(file_name, "r") as r:
            file_read = r.read()
            self.assertEqual(file_read, "Hello\n, world!")

    def test_wildcard_matching(self):
        """Test glob patterns"""
        for ext in [".txt", ".md"]:
            for i in range(3):
                file_path = self._path_to("wildcard/{}{}".format(i, ext))
                with tf.io.gfile.GFile(file_path, "w") as f:
                    f.write("")

        txt_files = tf.io.gfile.glob(self._path_to("wildcard/*.txt"))
        self.assertEqual(3, len(txt_files))
        for i, name in enumerate(txt_files):
            self.assertEqual(self._path_to("wildcard/{}.txt".format(i)), name)

        tf.io.gfile.rmtree(self._path_to("wildcard"))

    def test_delete_recursively(self):
        """Test delete recursively."""
        # Setup and check preconditions.
        dir_name = self._path_to("recursive")
        file_name = self._path_to("recursive/1")

        tf.io.gfile.mkdir(dir_name)
        with tf.io.gfile.GFile(file_name, "w") as w:
            w.write("")

        self.assertTrue(tf.io.gfile.isdir(dir_name))
        self.assertTrue(tf.io.gfile.exists(file_name))

        # Delete directory recursively.
        tf.io.gfile.rmtree(dir_name)

        # Check that directory was deleted.
        self.assertFalse(tf.io.gfile.exists(dir_name))
        self.assertFalse(tf.io.gfile.exists(file_name))

    def test_is_directory(self):
        """Test is directory."""
        # Setup and check preconditions.
        dir_name = self._path_to("isdir/1")
        file_name = self._path_to("isdir/2")
        with tf.io.gfile.GFile(file_name, "w") as w:
            w.write("")
        tf.io.gfile.mkdir(dir_name)
        # Check that directory is a directory.
        self.assertTrue(tf.io.gfile.isdir(dir_name))
        # Check that file is not a directory.
        self.assertFalse(tf.io.gfile.isdir(file_name))

    def test_list_directory(self):
        """Test list directory."""
        # Setup and check preconditions.
        dir_name = self._path_to("listdir")
        file_names = [self._path_to("listdir/{}".format(i)) for i in range(1, 4)]

        for file_name in file_names:
            with tf.io.gfile.GFile(file_name, "w") as w:
                w.write("")
        # Get list of files in directory.
        ls_result = tf.io.gfile.listdir(dir_name)
        # Check that list of files is correct.
        self.assertEqual(len(file_names), len(ls_result))
        for e in ["1", "2", "3"]:
            self.assertTrue(e in ls_result)

    def test_make_dirs(self):
        """Test make dirs."""
        # Setup and check preconditions.
        dir_name = self.path_root
        # Make directory.
        tf.io.gfile.mkdir(dir_name)
        # Check that directory was created.
        self.assertTrue(tf.io.gfile.isdir(dir_name))

        dir_name = self._path_to("test/directory")
        tf.io.gfile.mkdir(dir_name)
        self.assertTrue(tf.io.gfile.isdir(dir_name))

    def test_remove(self):
        """Test remove."""
        # Setup and check preconditions.
        file_name = self._path_to("1")
        self.assertFalse(tf.io.gfile.exists(file_name))
        with tf.io.gfile.GFile(file_name, "w") as w:
            w.write("")
        self.assertTrue(tf.io.gfile.exists(file_name))
        # Remove file.
        tf.io.gfile.remove(file_name)
        # Check that file was removed.
        self.assertFalse(tf.io.gfile.exists(file_name))

    def _test_read_file_offset_and_dataset(self):
        """Test read file with dataset"""
        # Note: disabled for now. Will enable once
        # all moved to eager mode
        # Setup and check preconditions.
        file_name = self._path_to("readfiledataset")
        if tf.io.gfile.exists(file_name):
            tf.io.gfile.remove(file_name)

        # Write data.
        with tf.io.gfile.GFile(file_name, "w") as w:
            w.write("Hello1,world1!\nHello2,world2!")
        dataset = tf.data.experimental.CsvDataset(file_name, [tf.string, tf.string])
        expected = [[b"Hello1", b"world1!"], [b"Hello2", b"world2!"]]
        i = 0
        for v in dataset:
            v0, v1 = v
            assert v0.numpy() == expected[i][0]
            assert v1.numpy() == expected[i][1]
            i += 1
        assert i == 2


if __name__ == "__main__":
    tf.test.main()
