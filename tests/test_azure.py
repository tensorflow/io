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

from tensorflow.python.platform import test
from tensorflow.python.platform import gfile
import tensorflow_io.azure.azfs_ops  # pylint: disable=unused-import

class AZFSTest(test.TestCase):
  """[summary]

  Arguments:
    test {[type]} -- [description]
  """

  def __init__(self, methodName='runTest'): # pylint: disable=invalid-name
    self.account = 'devstoreaccount1'
    self.container = 'aztest'
    self.path_root = 'az://' + os.path.join(self.account, self.container)
    super(AZFSTest, self).__init__(methodName)

  def _path_to(self, path):
    return os.path.join(self.path_root, path)

  def setUp(self): # pylint: disable=invalid-name
    super(AZFSTest, self).setUp()
    if not gfile.IsDirectory(self.path_root):
      gfile.MakeDirs(self.path_root)

  def test_exists(self):
    self.assertTrue(gfile.IsDirectory(self.path_root))

  def test_also_works_with_full_dns_name(self):
    """Test the file system also works when we're given
       a path of the form
       az://<account>.blob.core.windows.net/<container>/<path>
    """
    file_name = self.account + '.blob.core.windows.net' + self.container
    if not gfile.IsDirectory(file_name):
      gfile.MakeDirs(file_name)

    self.assertTrue(gfile.IsDirectory(file_name))

  def test_create_file(self):
    """Test create file.
    """
    # Setup and check preconditions.
    file_name = self._path_to("testfile")
    if gfile.Exists(file_name):
      gfile.Remove(file_name)
    # Create file.
    with gfile.Open(file_name, 'w') as w:
      w.write("")
    # Check that file was created.
    self.assertTrue(gfile.Exists(file_name))

    gfile.Remove(file_name)

  def test_write_read_file(self):
    """Test write/read file.
    """
    # Setup and check preconditions.
    file_name = self._path_to("writereadfile")
    if gfile.Exists(file_name):
      gfile.Remove(file_name)

    # Write data.
    with gfile.Open(file_name, 'w') as w:
      w.write("Hello\n, world!")

    # Read data.
    with gfile.Open(file_name, 'r') as r:
      file_read = r.read()
      self.assertEqual(file_read, "Hello\n, world!")

  def test_wildcard_matching(self):
    """Test glob patterns"""
    for ext in [".txt", ".md"]:
      for i in range(3):
        file_path = self._path_to("wildcard/{}{}".format(i, ext))
        with gfile.Open(file_path, 'w') as f:
          f.write('')

    txt_files = gfile.Glob(self._path_to("wildcard/*.txt"))
    self.assertEqual(3, len(txt_files))
    for i, name in enumerate(txt_files):
      self.assertEqual(self._path_to("wildcard/{}.txt".format(i)), name)

    gfile.DeleteRecursively(self._path_to("wildcard"))

  def test_delete_recursively(self):
    """Test delete recursively.
    """
    # Setup and check preconditions.
    dir_name = self._path_to("recursive")
    file_name = self._path_to("recursive/1")

    gfile.MkDir(dir_name)
    with gfile.Open(file_name, 'w') as w:
      w.write("")

    self.assertTrue(gfile.IsDirectory(dir_name))
    self.assertTrue(gfile.Exists(file_name))

    # Delete directory recursively.
    gfile.DeleteRecursively(dir_name)

    # Check that directory was deleted.
    self.assertFalse(gfile.Exists(dir_name))
    self.assertFalse(gfile.Exists(file_name))

  def test_is_directory(self):
    """Test is directory.

    """
    # Setup and check preconditions.
    dir_name = self._path_to("isdir/1")
    file_name = self._path_to("isdir/2")
    with gfile.Open(file_name, 'w') as w:
      w.write("")
    gfile.MkDir(dir_name)
    # Check that directory is a directory.
    self.assertTrue(gfile.IsDirectory(dir_name))
    # Check that file is not a directory.
    self.assertFalse(gfile.IsDirectory(file_name))

  def test_list_directory(self):
    """Test list directory.

    """
    # Setup and check preconditions.
    dir_name = self._path_to("listdir")
    file_names = [
        self._path_to("listdir/{}".format(i)) for i in range(1, 4)
    ]

    for file_name in file_names:
      with gfile.Open(file_name, 'w') as w:
        w.write("")
    # Get list of files in directory.
    ls_result = gfile.ListDirectory(dir_name)
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
    gfile.MkDir(dir_name)
    # Check that directory was created.
    self.assertTrue(gfile.IsDirectory(dir_name))

    dir_name = self._path_to("test/directory")
    gfile.MkDir(dir_name)
    self.assertTrue(gfile.IsDirectory(dir_name))

  def test_remove(self):
    """Test remove.
    """
    # Setup and check preconditions.
    file_name = self._path_to("1")
    self.assertFalse(gfile.Exists(file_name))
    with gfile.Open(file_name, 'w') as w:
      w.write("")
    self.assertTrue(gfile.Exists(file_name))
    # Remove file.
    gfile.Remove(file_name)
    # Check that file was removed.
    self.assertFalse(gfile.Exists(file_name))

if __name__ == '__main__':
  test.main()
