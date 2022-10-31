# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for OSS filesystem"""

import os
import unittest

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test
from tensorflow.python.platform import gfile
import tensorflow as tf
import tensorflow_io as tfio  # pylint: disable=unused-import

get_oss_path = None
access_id = os.environ.get("OSS_ACCESS_ID")
bucket = os.environ.get("OSS_BUCKET")
access_key = os.environ.get("OSS_ACCESS_KEY")
host = os.environ.get("OSS_HOST")

_msg = (
    "OSS tests skipped. To enable them, set the OSS-related " "environment variables."
)


def _check_oss_variable():
    return (
        access_id is not None
        and access_key is not None
        and host is not None
        and bucket is not None
    )


@unittest.skipIf(not _check_oss_variable(), _msg)
class OSSFSTest(test.TestCase):
    """OSS Filesystem Tests"""

    @classmethod
    def setUpClass(cls):  # pylint: disable=invalid-name
        global get_oss_path
        get_oss_path = lambda p: os.path.join(
            "oss://%s\x01id=%s\x02key=%s\x02host=%s"
            % (bucket, access_id, access_key, host),
            "oss_fs_test",
            p,
        )
        gfile.MkDir(get_oss_path(""))

    @classmethod
    def tearDownClass(cls):  # pylint: disable=invalid-name
        gfile.DeleteRecursively(get_oss_path(""))

    def setUp(self):
        self._base_dir = file_io.join(
            "oss://%s\x01id=%s\x02key=%s\x02host=%s"
            % (bucket, access_id, access_key, host),
            "base_dir",
        )
        file_io.create_dir(self._base_dir)

    def tearDown(self):
        file_io.delete_recursively(self._base_dir)

    def test_file_operations(self):
        """Test file operations"""

        f = get_oss_path("test_file_operations")
        self.assertFalse(gfile.Exists(f))

        fh = gfile.Open(f, mode="w")
        content = "file content"
        fh.write(content)
        fh.close()
        self.assertTrue(gfile.Exists(f))

        fh = gfile.Open(f)
        self.assertEqual(fh.read(), content)

        self.assertEqual(gfile.Stat(f).length, len(content))

        f2 = get_oss_path("test_file_2")
        gfile.Rename(f, f2)
        self.assertFalse(gfile.Exists(f))
        self.assertTrue(gfile.Exists(f2))

        f3 = get_oss_path("test_file_3")
        gfile.Copy(f2, f3, overwrite=True)
        self.assertTrue(gfile.Exists(f3))

    def test_dir_operations(self):
        """Test directory operations"""

        d = get_oss_path("d1/d2/d3/d4")
        gfile.MakeDirs(d)
        self.assertTrue(gfile.Stat(d).is_directory)

        # Test listing test directory with and without trailing '/'
        content = gfile.ListDirectory(
            "oss://%s\x01id=%s\x02key=%s\x02host=%s"
            % (bucket, access_id, access_key, host)
            + "/oss_fs_test"
        )
        content_s = gfile.ListDirectory(
            "oss://%s\x01id=%s\x02key=%s\x02host=%s"
            % (bucket, access_id, access_key, host)
            + "/oss_fs_test/"
        )
        self.assertEqual(content, content_s)
        self.assertIn("d1", content)

        # Test listing sub directories.
        content = gfile.ListDirectory(get_oss_path("d1"))
        content_s = gfile.ListDirectory(get_oss_path("d1/"))
        self.assertEqual(content, content_s)
        self.assertIn("d2", content)

        content = gfile.ListDirectory(get_oss_path("d1/d2/d3/d4"))
        content_s = gfile.ListDirectory(get_oss_path("d1/d2/d3/d4"))
        self.assertEqual(content, content_s)
        self.assertEqual([], content)

        # Test Rename directories
        self.assertTrue(gfile.Exists(get_oss_path("d1")))
        gfile.Rename(get_oss_path("d1"), get_oss_path("rename_d1"), overwrite=True)
        self.assertTrue(gfile.Exists(get_oss_path("rename_d1")))
        self.assertFalse(gfile.Exists(get_oss_path("d1")))

        content = gfile.ListDirectory(get_oss_path("rename_d1"))
        content_s = gfile.ListDirectory(get_oss_path("rename_d1/"))
        self.assertEqual(content, content_s)
        self.assertIn("d2", content)

        # Test Rename non-empty directories
        not_empty_dir = get_oss_path("not_empty_dir/")
        rename_not_empty_dir = get_oss_path("rename_not_empty_dir/")
        gfile.MakeDirs(not_empty_dir)
        not_empty_file = get_oss_path("not_empty_dir/not_empty_file")
        rename_not_empty_file = get_oss_path("rename_not_empty_dir/not_empty_file")
        with gfile.Open(not_empty_file, mode="w") as fh:
            content = "file content"
            fh.write(content)
        self.assertTrue(gfile.Exists(not_empty_dir))
        self.assertTrue(gfile.Exists(not_empty_file))
        gfile.Rename(not_empty_dir, rename_not_empty_dir, overwrite=True)
        self.assertFalse(gfile.Exists(not_empty_dir))
        self.assertFalse(gfile.Exists(not_empty_file))
        self.assertTrue(gfile.Exists(rename_not_empty_dir))
        self.assertTrue(gfile.Exists(rename_not_empty_file))

    def test_file_doesnt_exist(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        self.assertFalse(gfile.Exists(file_path))
        with self.assertRaises(errors.NotFoundError):
            _ = file_io.read_file_to_string(file_path)

    def test_write_to_string(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="w") as f:
            f.write("testing")
        self.assertTrue(gfile.Exists(file_path))
        file_contents = file_io.read_file_to_string(file_path)
        self.assertEqual("testing", file_contents)

    def test_append(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with self.assertRaises(errors.UnimplementedError):
            with gfile.Open(file_path, mode="a") as f:
                f.write("a1\n")

    def test_read_binary_mode(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        file_io.write_string_to_file(file_path, "testing")
        with gfile.Open(file_path, mode="rb") as f:
            self.assertEqual(b"testing", f.read())

    def test_write_binary_mode(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="wb") as f:
            f.write("testing")
        with gfile.Open(file_path, mode="r") as f:
            self.assertEqual("testing", f.read())

    def test_multiple_writes(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="w") as f:
            f.write("line1\n")
            f.write("line2")
        file_contents = file_io.read_file_to_string(file_path)
        self.assertEqual("line1\nline2", file_contents)

    def test_file_write_bad_mode(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with self.assertRaises(errors.PermissionDeniedError):
            gfile.Open(file_path, mode="r").write("testing")

    def test_file_read_bad_mode(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="wb") as f:
            f.write("testing")
        self.assertTrue(file_io.file_exists(file_path))
        with self.assertRaises(errors.PermissionDeniedError):
            gfile.Open(file_path, mode="w").read()

    def test_file_delete(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="w") as f:
            f.write("testing")
        gfile.Remove(file_path)
        self.assertFalse(gfile.Exists(file_path))

    def test_create_recursive_dir(self):
        dir_path = file_io.join(self._base_dir, "temp_dir/temp_dir1/temp_dir2")
        gfile.MakeDirs(dir_path)
        gfile.MakeDirs(dir_path)  # repeat creation
        file_path = file_io.join(str(dir_path), "temp_file")
        with gfile.Open(file_path, mode="w") as f:
            f.write("testing")
        self.assertTrue(gfile.Exists(file_path))
        gfile.DeleteRecursively(file_io.join(self._base_dir, "temp_dir"))
        self.assertFalse(gfile.Exists(file_path))

    def test_copy(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="w") as f:
            f.write("testing")
        copy_path = file_io.join(self._base_dir, "copy_file")
        gfile.Copy(file_path, copy_path)
        self.assertTrue(file_io.file_exists(copy_path))
        f = gfile.Open(file_path, mode="r")
        self.assertEqual("testing", f.read())
        self.assertEqual(7, f.tell())

    def test_copy_overwrite(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="w") as f:
            f.write("testing")
        copy_path = file_io.join(self._base_dir, "copy_file")
        gfile.Open(copy_path, mode="w").write("copy")
        gfile.Copy(file_path, copy_path, overwrite=True)
        self.assertTrue(gfile.Exists(copy_path))
        self.assertEqual("testing", gfile.Open(copy_path, mode="r").read())

    def test_copy_overwrite_false(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="w") as f:
            f.write("testing")
        copy_path = file_io.join(self._base_dir, "copy_file")
        with gfile.Open(copy_path, mode="w") as f:
            f.write("copy")
        with self.assertRaises(errors.AlreadyExistsError):
            gfile.Copy(file_path, copy_path, overwrite=False)

    def test_rename(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="w") as f:
            f.write("testing")
        rename_path = file_io.join(self._base_dir, "rename_file")
        gfile.Rename(file_path, rename_path)
        self.assertTrue(file_io.file_exists(rename_path))
        self.assertFalse(file_io.file_exists(file_path))

    def test_rename_overwrite(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="w") as f:
            f.write("testing")
        rename_path = file_io.join(self._base_dir, "rename_file")
        with gfile.Open(rename_path, mode="w") as f:
            f.write("rename")
        gfile.Rename(file_path, rename_path, overwrite=True)
        self.assertTrue(gfile.Exists(rename_path))
        self.assertFalse(gfile.Exists(file_path))

    def test_rename_overwrite_false(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="w") as f:
            f.write("testing")
        rename_path = file_io.join(self._base_dir, "rename_file")
        with gfile.Open(rename_path, mode="w") as f:
            f.write("rename")
        with self.assertRaises(errors.AlreadyExistsError):
            gfile.Rename(file_path, rename_path, overwrite=False)
        self.assertTrue(gfile.Exists(rename_path))
        self.assertTrue(gfile.Exists(file_path))

    def test_delete_recursively_fail(self):
        fake_dir_path = file_io.join(self._base_dir, "temp_dir")
        with self.assertRaises(errors.NotFoundError):
            gfile.DeleteRecursively(fake_dir_path)

    def test_is_directory(self):
        dir_path = file_io.join(self._base_dir, "test_dir")
        # Failure for a non-existing dir.
        self.assertFalse(gfile.IsDirectory(dir_path))
        gfile.MkDir(dir_path)
        self.assertTrue(gfile.IsDirectory(dir_path))
        file_path = file_io.join(str(dir_path), "test_file")
        with gfile.Open(file_path, mode="w") as f:
            f.write("testing")
        # False for a file.
        self.assertFalse(gfile.IsDirectory(file_path))
        # Test that the value returned from `stat()` has `is_directory` set.
        file_statistics = gfile.Stat(dir_path)
        self.assertTrue(file_statistics.is_directory)

    def test_list_directory(self):
        dir_path = file_io.join(self._base_dir, "test_dir")
        gfile.MkDir(dir_path)
        files = ["file1.txt", "file2.txt", "file3.txt"]
        for name in files:
            file_path = file_io.join(str(dir_path), name)
            with gfile.Open(file_path, mode="w") as f:
                f.write("testing")
        subdir_path = file_io.join(str(dir_path), "sub_dir")
        gfile.MkDir(subdir_path)
        subdir_file_path = file_io.join(str(subdir_path), "file4.txt")
        with gfile.Open(subdir_file_path, mode="w") as f:
            f.write("testing")
        dir_list = gfile.ListDirectory(dir_path)
        self.assertItemsEqual(files + ["sub_dir"], dir_list)

    def test_list_directory_failure(self):
        dir_path = file_io.join(self._base_dir, "test_dir")
        with self.assertRaises(errors.NotFoundError):
            gfile.ListDirectory(dir_path)

    def _setup_walk_directories(self, dir_path):
        # Creating a file structure as follows
        # test_dir -> file: file1.txt; dirs: subdir1_1, subdir1_2, subdir1_3
        # subdir1_1 -> file: file3.txt
        # subdir1_2 -> dir: subdir2
        gfile.MkDir(dir_path)
        with gfile.Open(file_io.join(dir_path, "file1.txt"), mode="w") as f:
            f.write("testing")
        sub_dirs1 = ["subdir1_1", "subdir1_2", "subdir1_3"]
        for name in sub_dirs1:
            gfile.MkDir(file_io.join(dir_path, name))
        with gfile.Open(file_io.join(dir_path, "subdir1_1/file2.txt"), mode="w") as f:
            f.write("testing")
        gfile.MkDir(file_io.join(dir_path, "subdir1_2/subdir2"))

    def test_walk_in_order(self):
        dir_path_str = file_io.join(self._base_dir, "test_dir")
        dir_path = file_io.join(self._base_dir, "test_dir")
        self._setup_walk_directories(dir_path_str)
        # Now test the walk (in_order = True)
        all_dirs = []
        all_subdirs = []
        all_files = []
        for (w_dir, w_subdirs, w_files) in file_io.walk(dir_path, in_order=True):
            all_dirs.append(w_dir)
            all_subdirs.append(w_subdirs)
            all_files.append(w_files)
        self.assertItemsEqual(
            all_dirs,
            [dir_path_str]
            + [
                file_io.join(dir_path_str, item)
                for item in ["subdir1_1", "subdir1_2", "subdir1_2/subdir2", "subdir1_3"]
            ],
        )
        self.assertEqual(dir_path_str, all_dirs[0])
        self.assertLess(
            all_dirs.index(file_io.join(dir_path_str, "subdir1_2")),
            all_dirs.index(file_io.join(dir_path_str, "subdir1_2/subdir2")),
        )
        self.assertItemsEqual(all_subdirs[1:5], [[], ["subdir2"], [], []])
        self.assertItemsEqual(all_subdirs[0], ["subdir1_1", "subdir1_2", "subdir1_3"])
        self.assertItemsEqual(all_files, [["file1.txt"], ["file2.txt"], [], [], []])
        self.assertLess(all_files.index(["file1.txt"]), all_files.index(["file2.txt"]))

    def test_walk_post_order(self):
        dir_path = file_io.join(self._base_dir, "test_dir")
        self._setup_walk_directories(dir_path)
        # Now test the walk (in_order = False)
        all_dirs = []
        all_subdirs = []
        all_files = []
        for (w_dir, w_subdirs, w_files) in file_io.walk(dir_path, in_order=False):
            all_dirs.append(w_dir)
            all_subdirs.append(w_subdirs)
            all_files.append(w_files)
        self.assertItemsEqual(
            all_dirs,
            [
                file_io.join(dir_path, item)
                for item in ["subdir1_1", "subdir1_2/subdir2", "subdir1_2", "subdir1_3"]
            ]
            + [dir_path],
        )
        self.assertEqual(dir_path, all_dirs[4])
        self.assertLess(
            all_dirs.index(file_io.join(dir_path, "subdir1_2/subdir2")),
            all_dirs.index(file_io.join(dir_path, "subdir1_2")),
        )
        self.assertItemsEqual(all_subdirs[0:4], [[], [], ["subdir2"], []])
        self.assertItemsEqual(all_subdirs[4], ["subdir1_1", "subdir1_2", "subdir1_3"])
        self.assertItemsEqual(all_files, [["file2.txt"], [], [], [], ["file1.txt"]])
        self.assertLess(all_files.index(["file2.txt"]), all_files.index(["file1.txt"]))

    def test_walk_failure(self):
        dir_path = file_io.join(self._base_dir, "test_dir")
        # Try walking a directory that wasn't created.
        all_dirs = []
        all_subdirs = []
        all_files = []
        for (w_dir, w_subdirs, w_files) in file_io.walk(dir_path, in_order=False):
            all_dirs.append(w_dir)
            all_subdirs.append(w_subdirs)
            all_files.append(w_files)
        self.assertItemsEqual(all_dirs, [])
        self.assertItemsEqual(all_subdirs, [])
        self.assertItemsEqual(all_files, [])

    def test_stat(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="w") as f:
            f.write("testing")
        file_statistics = gfile.Stat(file_path)
        self.assertEqual(7, file_statistics.length)
        self.assertFalse(file_statistics.is_directory)

    def test_read_line(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="r+") as f:
            f.write("testing1\ntesting2\ntesting3\n\ntesting5")
        self.assertEqual(36, f.size())
        self.assertEqual("testing1\n", f.readline())
        self.assertEqual("testing2\n", f.readline())
        self.assertEqual("testing3\n", f.readline())
        self.assertEqual("\n", f.readline())
        self.assertEqual("testing5", f.readline())
        self.assertEqual("", f.readline())

    def test_read(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="r+") as f:
            f.write("testing1\ntesting2\ntesting3\n\ntesting5")
        self.assertEqual(36, f.size())
        self.assertEqual("testing1\n", f.read(9))
        self.assertEqual("testing2\n", f.read(9))
        self.assertEqual("t", f.read(1))
        self.assertEqual("esting3\n\ntesting5", f.read())

    def test_read_error_reacquires_gil(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="r+") as f:
            f.write("testing1\ntesting2\ntesting3\n\ntesting5")
        with self.assertRaises(errors.InvalidArgumentError):
            # At present, this is sufficient to convince ourselves that the change
            # fixes the problem. That is, this test will seg fault without the change,
            # and pass with it. Unfortunately, this is brittle, as it relies on the
            # Python layer to pass the argument along to the wrapped C++ without
            # checking the argument itself.
            f.read(-2)

    def test_tell(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="r+") as f:
            f.write("testing1\ntesting2\ntesting3\n\ntesting5")
        self.assertEqual(0, f.tell())
        self.assertEqual("testing1\n", f.readline())
        self.assertEqual(9, f.tell())
        self.assertEqual("testing2\n", f.readline())
        self.assertEqual(18, f.tell())
        self.assertEqual("testing3\n", f.readline())
        self.assertEqual(27, f.tell())
        self.assertEqual("\n", f.readline())
        self.assertEqual(28, f.tell())
        self.assertEqual("testing5", f.readline())
        self.assertEqual(36, f.tell())
        self.assertEqual("", f.readline())
        self.assertEqual(36, f.tell())

    def test_seek(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="r+") as f:
            f.write("testing1\ntesting2\ntesting3\n\ntesting5")
        self.assertEqual("testing1\n", f.readline())
        self.assertEqual(9, f.tell())

        # Seek to 18
        f.seek(18)
        self.assertEqual(18, f.tell())
        self.assertEqual("testing3\n", f.readline())

        # Seek back to 9
        f.seek(9)
        self.assertEqual(9, f.tell())
        self.assertEqual("testing2\n", f.readline())

        f.seek(0)
        self.assertEqual(0, f.tell())
        self.assertEqual("testing1\n", f.readline())

        with self.assertRaises(errors.InvalidArgumentError):
            f.seek(-1)

        with self.assertRaises(TypeError):
            f.seek()

        with self.assertRaises(TypeError):
            f.seek(offset=0, position=0)
        f.seek(position=9)
        self.assertEqual(9, f.tell())
        self.assertEqual("testing2\n", f.readline())

    def test_seek_from_what(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        with gfile.Open(file_path, mode="r+") as f:
            f.write("testing1\ntesting2\ntesting3\n\ntesting5")
        self.assertEqual("testing1\n", f.readline())
        self.assertEqual(9, f.tell())

        # Seek to 18
        f.seek(9, 1)
        self.assertEqual(18, f.tell())
        self.assertEqual("testing3\n", f.readline())

        # Seek back to 9
        f.seek(9, 0)
        self.assertEqual(9, f.tell())
        self.assertEqual("testing2\n", f.readline())

        f.seek(-f.size(), 2)
        self.assertEqual(0, f.tell())
        self.assertEqual("testing1\n", f.readline())

        with self.assertRaises(errors.InvalidArgumentError):
            f.seek(0, 3)

    def test_reading_iterator(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        data = ["testing1\n", "testing2\n", "testing3\n", "\n", "testing5"]
        with gfile.Open(file_path, mode="r+") as f:
            f.write("".join(data))
        actual_data = []
        for line in f:
            actual_data.append(line)
        self.assertSequenceEqual(actual_data, data)

    def test_read_lines(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        data = ["testing1\n", "testing2\n", "testing3\n", "\n", "testing5"]
        f = gfile.Open(file_path, mode="r+")
        f.write("".join(data))
        f.flush()
        f.close()
        lines = f.readlines()
        self.assertSequenceEqual(lines, data)

    def test_utf8_string_path(self):
        file_path = file_io.join(self._base_dir, "UTF8测试_file")
        file_io.write_string_to_file(file_path, "testing")
        self.assertTrue(gfile.Exists(file_path))
        with gfile.Open(file_path, mode="rb") as f:
            self.assertEqual(b"testing", f.read())

    def test_eof(self):
        file_path = file_io.join(self._base_dir, "temp_file")
        f = gfile.Open(file_path, mode="r+")
        content = "testing"
        f.write(content)
        f.flush()
        f.close()
        self.assertEqual(content, f.read(len(content) + 1))


if __name__ == "__main__":
    test.main()
