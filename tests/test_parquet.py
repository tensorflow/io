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
"""Tests for read_parquet and ParquetDataset."""


import os
import collections
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio

import pandas as pd

filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_parquet",
    "parquet_cpp_example.parquet",
)
filename = "file://" + filename

# Note: The sample file is generated from the following after apply patch
# tests/test_parquet/parquet_cpp_example.patch:
# `parquet-cpp/examples/low-level-api/reader_writer`
# This test extracts columns of [0, 1, 2, 4, 5]
# with column data types of [bool, int32, int64, float, double].
# Please check `parquet-cpp/examples/low-level-api/reader-writer.cc`
# to find details of how records are generated:
# Column 0 (bool): True for even rows and False otherwise.
# Column 1 (int32): Equal to row_index.
# Column 2 (int64): Equal to row_index * 1000 * 1000 * 1000 * 1000.
# Column 4 (float): Equal to row_index * 1.1.
# Column 5 (double): Equal to row_index * 1.1111111.
def test_parquet():
    """Test case for read_parquet."""
    parquet = tfio.IOTensor.from_parquet(filename)
    columns = [
        "boolean_field",
        "int32_field",
        "int64_field",
        "int96_field",
        "float_field",
        "double_field",
        "ba_field",
        "flba_field",
    ]
    assert parquet.columns == columns
    p0 = parquet("boolean_field")
    p1 = parquet("int32_field")
    p2 = parquet("int64_field")
    p4 = parquet("float_field")
    p5 = parquet("double_field")
    p6 = parquet("ba_field")
    p7 = parquet("flba_field")
    assert p0.dtype == tf.bool
    assert p1.dtype == tf.int32
    assert p2.dtype == tf.int64
    assert p4.dtype == tf.float32
    assert p5.dtype == tf.float64
    assert p6.dtype == tf.string
    assert p7.dtype == tf.string
    for i in range(500):  # 500 rows.
        v0 = (i % 2) == 0
        v1 = i
        v2 = i * 1000 * 1000 * 1000 * 1000
        v4 = 1.1 * i
        v5 = 1.1111111 * i
        v6 = b"parquet%03d" % i
        v7 = bytearray(b"").join([bytearray((i % 256,)) for _ in range(10)])
        assert v0 == p0[i].numpy()
        assert v1 == p1[i].numpy()
        assert v2 == p2[i].numpy()
        assert np.isclose(v4, p4[i].numpy())
        assert np.isclose(v5, p5[i].numpy())
        assert v6 == p6[i].numpy()
        assert v7 == p7[i].numpy()

    # test parquet dataset
    columns = [
        "boolean_field",
        "int32_field",
        "int64_field",
        "float_field",
        "double_field",
        "ba_field",
        "flba_field",
    ]
    dataset = tfio.IODataset.from_parquet(filename, columns)
    i = 0
    for v in dataset:
        v0 = (i % 2) == 0
        v1 = i
        v2 = i * 1000 * 1000 * 1000 * 1000
        v4 = 1.1 * i
        v5 = 1.1111111 * i
        v6 = b"parquet%03d" % i
        v7 = bytearray(b"").join([bytearray((i % 256,)) for _ in range(10)])
        p0 = v["boolean_field"]
        p1 = v["int32_field"]
        p2 = v["int64_field"]
        p4 = v["float_field"]
        p5 = v["double_field"]
        p6 = v["ba_field"]
        p7 = v["flba_field"]
        assert v0 == p0.numpy()
        assert v1 == p1.numpy()
        assert v2 == p2.numpy()
        assert np.isclose(v4, p4.numpy())
        assert np.isclose(v5, p5.numpy())
        assert v6 == p6.numpy()
        assert v7 == p7.numpy()
        i += 1


def test_parquet_dataset_ordered_dict():
    """Test case for order and dict of parquet dataset"""
    parquet = tfio.IODataset.from_parquet(filename)
    assert parquet.element_spec == collections.OrderedDict(
        [
            (b"boolean_field", tf.TensorSpec(shape=(), dtype=tf.bool)),
            (b"int32_field", tf.TensorSpec(shape=(), dtype=tf.int32)),
            (b"int64_field", tf.TensorSpec(shape=(), dtype=tf.int64)),
            (b"int96_field", tf.TensorSpec(shape=(), dtype=tf.int64)),
            (b"float_field", tf.TensorSpec(shape=(), dtype=tf.float32)),
            (b"double_field", tf.TensorSpec(shape=(), dtype=tf.float64)),
            (b"ba_field", tf.TensorSpec(shape=(), dtype=tf.string)),
            (b"flba_field", tf.TensorSpec(shape=(), dtype=tf.string)),
        ]
    )


def test_parquet_graph():
    """Test case for parquet in graph mode."""

    # test parquet dataset
    @tf.function(autograph=False)
    def f(e):
        columns = {
            "boolean_field": tf.bool,
            "int32_field": tf.int32,
            "int64_field": tf.int64,
            "float_field": tf.float32,
            "double_field": tf.float64,
            "ba_field": tf.string,
            "flba_field": tf.string,
        }
        dataset = tfio.IODataset.from_parquet(e, columns)
        dataset = dataset.batch(500)
        return tf.data.experimental.get_single_element(dataset)

    data = f(filename)

    for i in range(500):
        v0 = (i % 2) == 0
        v1 = i
        v2 = i * 1000 * 1000 * 1000 * 1000
        v4 = 1.1 * i
        v5 = 1.1111111 * i
        v6 = b"parquet%03d" % i
        v7 = bytearray(b"").join([bytearray((i % 256,)) for _ in range(10)])
        p0 = data["boolean_field"][i]
        p1 = data["int32_field"][i]
        p2 = data["int64_field"][i]
        p4 = data["float_field"][i]
        p5 = data["double_field"][i]
        p6 = data["ba_field"][i]
        p7 = data["flba_field"][i]
        assert v0 == p0.numpy()
        assert v1 == p1.numpy()
        assert v2 == p2.numpy()
        assert np.isclose(v4, p4.numpy())
        assert np.isclose(v5, p5.numpy())
        assert v6 == p6.numpy()
        assert v7 == p7.numpy()


def test_parquet_data():
    """Test case for parquet GitHub 1254"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_parquet",
        "part-00000-ca0e89bf-ccd7-47e1-925c-9b42c8716c84-c000.snappy.parquet",
    )
    parquet = pd.read_parquet(filename)
    dataset = tfio.IODataset.from_parquet(filename)
    i = 0
    for columns in dataset:
        assert columns[b"user_id"] == parquet["user_id"][i]
        assert columns[b"movie_id"] == parquet["movie_id"][i]
        assert columns[b"movie_title"] == parquet["movie_title"][i]
        assert columns[b"rating"] == parquet["rating"][i]
        assert columns[b"timestamp"] == parquet["timestamp"][i]
        i += 1


def test_parquet_dataset_from_file_pattern():
    """Test the parquet dataset creation process using a file pattern"""
    df = pd.DataFrame({"pred_0": 0.1 * np.arange(100), "pred_1": -0.1 * np.arange(100)})
    df.to_parquet("df_x.parquet")
    df.to_parquet("df_y.parquet")
    columns = {
        "pred_0": tf.TensorSpec(tf.TensorShape([]), tf.double),
        "pred_1": tf.TensorSpec(tf.TensorShape([]), tf.double),
    }

    def map_fn(file_location):
        return tfio.IODataset.from_parquet(file_location, columns=columns)

    ds = tf.data.Dataset.list_files("*.parquet").map(map_fn)

    for d in ds:  # loop over the files
        for item in d:  # loop over items
            print(item)


def test_parquet_dataset_from_dir_failure():
    """Test the dataset creation failure when a directory is passed
    instead of a filename."""
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_parquet")
    columns = {
        "pred_0": tf.TensorSpec(tf.TensorShape([]), tf.double),
        "pred_1": tf.TensorSpec(tf.TensorShape([]), tf.double),
    }
    try:
        _ = tfio.IODataset.from_parquet(dir_path, columns=columns)
    except Exception as e:
        assert (
            str(e)
            == "passing a directory path to 'filename' is not supported. "
            + "Use 'tf.data.Dataset.list_files()' with a map() operation instead. "
            + "[Op:IO>ParquetReadableInfo]"
        )


if __name__ == "__main__":
    test.main()
