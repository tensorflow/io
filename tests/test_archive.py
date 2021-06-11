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
"""Tests for read_archive."""


import os
import tensorflow as tf
import tensorflow_io.python.ops.archive_ops as archive_io


def test_gz():
    """test_archive"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_parquet",
        "parquet_cpp_example.parquet.gz",
    )
    filename = "file://" + filename

    (
        format,  # pylint: disable=redefined-builtin
        entries,
    ) = archive_io.list_archive_entries(filename, ["gz", "tar.gz"])
    assert format.numpy().decode() == "gz"
    assert entries.shape == [1]

    elements = archive_io.read_archive(filename, format, entries)
    assert elements.shape == [1]

    expected_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_parquet",
        "parquet_cpp_example.parquet",
    )
    expected_filename = "file://" + expected_filename

    assert elements[0].numpy() == tf.io.read_file(expected_filename).numpy()


def test_none():
    """test_none"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_parquet",
        "parquet_cpp_example.parquet",
    )
    filename = "file://" + filename

    (
        format,  # pylint: disable=redefined-builtin
        entries,
    ) = archive_io.list_archive_entries(filename, ["none", "gz"])
    assert format.numpy().decode() == "none"
    assert entries.shape == [1]

    elements = archive_io.read_archive(filename, format, entries)
    assert elements.shape == [1]

    assert elements[0].numpy() == tf.io.read_file(filename).numpy()


def test_tar_gz():
    """test_tar_gz"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_parquet",
        "parquet_cpp_example.parquet.tar.gz",
    )
    filename = "file://" + filename

    (
        format,  # pylint: disable=redefined-builtin
        entries,
    ) = archive_io.list_archive_entries(filename, ["gz", "tar.gz"])
    assert format.numpy().decode() == "tar.gz"
    assert entries.shape == [2]
    assert entries[0].numpy().decode() == "parquet_cpp_example.parquet.1"
    assert entries[1].numpy().decode() == "parquet_cpp_example.parquet.2"

    elements = archive_io.read_archive(filename, format, entries)
    assert elements.shape == [2]

    expected_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_parquet",
        "parquet_cpp_example.parquet",
    )
    expected_filename = "file://" + expected_filename

    assert elements[0].numpy() == tf.io.read_file(expected_filename).numpy()
    assert elements[1].numpy() == tf.io.read_file(expected_filename).numpy()


def test_dataset():
    """test_dataset"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_parquet",
        "parquet_cpp_example.parquet.tar.gz",
    )
    filename = "file://" + filename

    # This is a demo implementation of ArchiveDataset
    dataset = (
        tf.compat.v2.data.Dataset.from_tensor_slices([filename])
        .map(lambda f: () + (f,) + archive_io.list_archive_entries(f, "tar.gz"))
        .map(
            lambda f, format, e: (
                tf.broadcast_to(f, tf.shape(e)),
                tf.broadcast_to(format, tf.shape(e)),
                e,
            )
        )
        .apply(tf.data.experimental.unbatch())
        .map(archive_io.read_archive)
    )

    expected_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_parquet",
        "parquet_cpp_example.parquet",
    )
    expected_filename = "file://" + expected_filename

    i = 0
    for entry in dataset:
        assert entry.numpy() == tf.io.read_file(expected_filename).numpy()
        i += 1
    assert i == 2


if __name__ == "__main__":
    test.main()
