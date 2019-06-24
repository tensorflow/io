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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow import errors                    # pylint: disable=wrong-import-position
from tensorflow import test                      # pylint: disable=wrong-import-position
from tensorflow.compat.v1 import data            # pylint: disable=wrong-import-position

import tensorflow_io.azure.azfs_ops # pylint: disable=unused-import

class AZFSTest(test.TestCase):
  """[summary]

  Arguments:
    test {[type]} -- [description]
  """

  def __init__(self, methodName='runTest'): # pylint: disable=invalid-name
    self.account = os.environ.get(
        'TF_AZURE_STORAGE_ACCOUNT', 'devstoreaccount1')
    self.container = os.environ.get('TF_AZURE_STORAGE_CONTAINER', 'aztest')
    self.path_root = 'az://' + os.path.join(self.account, self.container)
    super(AZFSTest, self).__init__(methodName)

  def _path_to(self, path):
    return os.path.join(self.path_root, path)

  def setUp(self): # pylint: disable=invalid-name
    super(AZFSTest, self).setUp()
    if not tf.gfile.IsDirectory(self.path_root):
      tf.gfile.MakeDirs(self.path_root)

  def tearDown(self): # pylint: disable=invalid-name
    #   Cleanup container
    if tf.gfile.IsDirectory(self.path_root):
      tf.gfile.DeleteRecursively(self.path_root)

  def test_exists(self):
    self.assertTrue(tf.gfile.IsDirectory(self.path_root))

  def test_also_works_with_full_dns_name(self):
    """Test the file system also works when we're given
       a path of the form
       az://<account>.blob.core.windows.net/<container>/<path>
    """
    file_name = self.account + '.blob.core.windows.net' + self.container
    if not tf.gfile.IsDirectory(file_name):
      tf.gfile.MakeDirs(file_name)

    self.assertTrue(tf.gfile.IsDirectory(file_name))

  def test_create_file(self):
    """Test create file.
    """
    # Setup and check preconditions.
    file_name = self._path_to("testfile")
    if tf.gfile.Exists(file_name):
      tf.gfile.Remove(file_name)
    # Create file.
    with tf.gfile.Open(file_name, 'w') as w:
      w.write("")
    # Check that file was created.
    self.assertTrue(tf.gfile.Exists(file_name))

    tf.gfile.Remove(file_name)

  def test_write_read_file(self):
    """Test write/read file.
    """
    # Setup and check preconditions.
    file_name = self._path_to("writereadfile")
    if tf.gfile.Exists(file_name):
      tf.gfile.Remove(file_name)

    # Write data.
    with tf.gfile.Open(file_name, 'w') as w:
      w.write("Hello\n, world!")

    # Read data.
    with tf.gfile.Open(file_name, 'r') as r:
      file_read = r.read()
      self.assertEqual(file_read, "Hello\n, world!")

  def test_wildcard_matching(self):
    """Test glob patterns"""
    for ext in [".txt", ".md"]:
      for i in range(3):
        file_path = self._path_to("wildcard/{}{}".format(i, ext))
        with tf.gfile.Open(file_path, 'w') as f:
          f.write('')

    txt_files = tf.gfile.Glob(self._path_to("wildcard/*.txt"))
    self.assertEqual(3, len(txt_files))
    for i, name in enumerate(txt_files):
      self.assertEqual(self._path_to("wildcard/{}.txt".format(i)), name)

    tf.gfile.DeleteRecursively(self._path_to("wildcard"))

  def test_delete_recursively(self):
    """Test delete recursively.
    """
    # Setup and check preconditions.
    dir_name = self._path_to("recursive")
    file_name = self._path_to("recursive/1")

    tf.gfile.MkDir(dir_name)
    with tf.gfile.Open(file_name, 'w') as w:
      w.write("")

    self.assertTrue(tf.gfile.IsDirectory(dir_name))
    self.assertTrue(tf.gfile.Exists(file_name))

    # Delete directory recursively.
    tf.gfile.DeleteRecursively(dir_name)

    # Check that directory was deleted.
    self.assertFalse(tf.gfile.Exists(dir_name))
    self.assertFalse(tf.gfile.Exists(file_name))

  def test_is_directory(self):
    """Test is directory.

    """
    # Setup and check preconditions.
    dir_name = self._path_to("isdir/1")
    file_name = self._path_to("isdir/2")
    with tf.gfile.Open(file_name, 'w') as w:
      w.write("")
    tf.gfile.MkDir(dir_name)
    # Check that directory is a directory.
    self.assertTrue(tf.gfile.IsDirectory(dir_name))
    # Check that file is not a directory.
    self.assertFalse(tf.gfile.IsDirectory(file_name))

  def test_list_directory(self):
    """Test list directory.

    """
    # Setup and check preconditions.
    dir_name = self._path_to("listdir")
    file_names = [
        self._path_to("listdir/{}".format(i)) for i in range(1, 4)
    ]

    for file_name in file_names:
      with tf.gfile.Open(file_name, 'w') as w:
        w.write("")
    # Get list of files in directory.
    ls_result = tf.gfile.ListDirectory(dir_name)
    # Check that list of files is correct.
    self.assertEqual(len(file_names), len(ls_result))
    for e in ["1", "2", "3"]:
      self.assertTrue(e in ls_result)

  def test_make_dirs(self):
    """Test make dirs.
    """
    # Setup and check preconditions.
    dir_name = self.path_root
    # Make directory.
    tf.gfile.MkDir(dir_name)
    # Check that directory was created.
    self.assertTrue(tf.gfile.IsDirectory(dir_name))

    dir_name = self._path_to("test/directory")
    tf.gfile.MkDir(dir_name)
    self.assertTrue(tf.gfile.IsDirectory(dir_name))

  def test_remove(self):
    """Test remove.
    """
    # Setup and check preconditions.
    file_name = self._path_to("1")
    self.assertFalse(tf.gfile.Exists(file_name))
    with tf.gfile.Open(file_name, 'w') as w:
      w.write("")
    self.assertTrue(tf.gfile.Exists(file_name))
    # Remove file.
    tf.gfile.Remove(file_name)
    # Check that file was removed.
    self.assertFalse(tf.gfile.Exists(file_name))

if __name__ == '__main__':
  test.main()
