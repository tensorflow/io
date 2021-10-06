import os
import sys
import pytest

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.python.lib.io import _pywrap_file_io
from tensorflow.python.util import compat

if sys.platform in ["darwin", "win32"]:
    pytest.skip("Incompatible", allow_module_level=True)


class DFSTest(tf.test.TestCase):

    def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name

        self.pool = os.environ["POOL_ID"]
        self.container = os.environ["CONT_ID"]
        self.path_root = "dfs://" + os.path.join(self.pool, self.container)
        super().__init__(methodName)

    def _path_to(self, path):
        return os.path.join(self.path_root, path)
    
    def _test_exists(self):
        self.assertTrue(tf.io.gfile.isdir(self.path_root))

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
        dir_name = self._path_to("wildcard")
        tf.io.gfile.mkdir(dir_name)
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
        parent = self._path_to("isdir")
        dir_name = self._path_to("isdir/1")
        file_name = self._path_to("isdir/5.txt")
        tf.io.gfile.mkdir(parent)
        with tf.io.gfile.GFile(file_name, "w") as w:
            w.write("123")
        tf.io.gfile.mkdir(dir_name)
        # Check that directory is a directory.
        self.assertTrue(tf.io.gfile.isdir(dir_name))
        # Check that file is not a directory.
        self.assertFalse(tf.io.gfile.isdir(file_name))
    
    def test_list_directory(self):
        """Test list directory."""
        # Setup and check preconditions.
        dir_name = self._path_to("listdir")
        tf.io.gfile.mkdir(dir_name)
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

        parent = self._path_to("test")
        dir_name = self._path_to("test/directory")
        tf.io.gfile.mkdir(parent)
        tf.io.gfile.makedirs(dir_name)
        self.assertTrue(tf.io.gfile.isdir(dir_name))

    def test_remove(self):
        """Test remove."""
        # Setup and check preconditions.
        file_name = self._path_to("file_to_be_removed")
        self.assertFalse(tf.io.gfile.exists(file_name))
        with tf.io.gfile.GFile(file_name, "w") as w:
            w.write("")
        self.assertTrue(tf.io.gfile.exists(file_name))
        # Remove file.
        tf.io.gfile.remove(file_name)
        # Check that file was removed.
        self.assertFalse(tf.io.gfile.exists(file_name))
    
if __name__ == "__main__":
    tf.test.main()