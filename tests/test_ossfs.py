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

from tensorflow.python.platform import test
from tensorflow.python.platform import gfile
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

        # Test listing bucket directory with and without trailing '/'
        content = gfile.ListDirectory(
            "oss://%s\x01id=%s\x02key=%s\x02host=%s"
            % (bucket, access_id, access_key, host)
        )
        content_s = gfile.ListDirectory(
            "oss://%s\x01id=%s\x02key=%s\x02host=%s/"
            % (bucket, access_id, access_key, host)
        )
        self.assertEqual(content, content_s)
        self.assertIn("oss_fs_test", content)
        self.assertIn("oss_fs_test/d1", content)
        self.assertIn("oss_fs_test/d1/d2", content)

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
        self.assertIn("d1/d2", content)

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


if __name__ == "__main__":
    test.main()
