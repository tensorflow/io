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
"""Tests for HDF5 file"""

import os
import glob
import shutil
import tempfile
import numpy as np
import h5py
import pytest

import tensorflow as tf
import tensorflow_io as tfio


def test_hdf5():
    """test_hdf5: GitHub issue 841"""

    def create_datasets(runpath, cnt=10):
        os.makedirs(runpath, exist_ok=True)
        for i in range(cnt):
            f = h5py.File(f"{runpath}/file_{i}.h5", "w")
            total_samples = np.random.randint(50000, 100000)
            f.create_dataset("features", data=np.random.random((total_samples, 60)))
            f.create_dataset("targets", data=np.random.random((total_samples, 3)))
            f.close()

    runpath = tempfile.mkdtemp()
    create_datasets(runpath)

    for i in range(2):
        cnt = 0
        for p in glob.glob(f"{runpath}/*.h5"):
            try:
                features = tfio.IODataset.from_hdf5(p, "/features")
                targets = tfio.IODataset.from_hdf5(p, "/targets")
                dataset = tf.data.Dataset.zip((features, targets))

                for t in dataset:
                    cnt += t[0].shape[0]

            except Exception as e:
                print(f"Failed going through {p}")
                raise e
            print(f"Success going through {p}")

    print(f"Iterated {cnt} items")

    shutil.rmtree(runpath)


def test_hdf5_grouped():
    """test_hdf5 with grouped data: https://github.com/tensorflow/io/issues/1161"""

    def create_datasets(runpath, cnt=10):
        os.makedirs(runpath, exist_ok=True)
        for i in range(cnt):
            f = h5py.File(f"{runpath}/file_{i}.h5", "w")
            total_samples = np.random.randint(50000, 100000)
            grp = f.create_group("sample_group")
            grp.create_dataset("features", data=np.random.random((total_samples, 60)))
            grp.create_dataset("targets", data=np.random.random((total_samples, 3)))
            f.close()

    runpath = tempfile.mkdtemp()
    create_datasets(runpath)

    for i in range(2):
        cnt = 0
        for p in glob.glob(f"{runpath}/*.h5"):
            try:
                features = tfio.IODataset.from_hdf5(p, "/sample_group/features")
                targets = tfio.IODataset.from_hdf5(p, "/sample_group/targets")
                dataset = tf.data.Dataset.zip((features, targets))

                for t in dataset:
                    cnt += t[0].shape[0]

            except Exception as e:
                print(f"Failed going through {p}")
                raise e
            print(f"Success going through {p}")

    print(f"Iterated {cnt} items")

    shutil.rmtree(runpath)


def test_hdf5_graph():
    """test_hdf5_graph: GitHub issue 898"""

    def create_datasets(runpath, cnt=10):
        filenames = [f"{runpath}/file_{i}.h5" for i in range(cnt)]
        samples = [np.random.randint(50000, 100000) for _ in range(cnt)]
        os.makedirs(runpath, exist_ok=True)
        for filename, sample in zip(filenames, samples):
            f = h5py.File(filename, "w")
            f.create_dataset("features", data=np.random.random((sample, 60)))
            f.create_dataset("targets", data=np.random.random((sample, 3)))
            f.close()
        return filenames, samples

    runpath = tempfile.mkdtemp()
    filenames, samples = create_datasets(runpath)

    @tf.function(autograph=False)
    def f(filename):
        spec = {"/features": tf.float64, "/targets": tf.float64}
        hdf5 = tfio.IOTensor.from_hdf5(filename, spec=spec)
        return tf.shape(hdf5("/features").to_tensor())[0]

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(f, num_parallel_calls=4)

    entries = [entry.numpy() for entry in dataset]

    print("Iterated items")
    for filename in filenames:
        print(f"File: {filename}")
    print(f"Samples: {samples}")
    print(f"Entries: {entries}")
    assert np.array_equal(entries, samples)

    shutil.rmtree(runpath)


def test_hdf5_bool():
    """test_hdf5_bool: GitHub issue 1144"""
    runpath = tempfile.mkdtemp()

    boolean_data = np.asarray(
        [True, False, True, False, True, False, True, False, True, False]
    )

    with h5py.File(f"{runpath}/my_data.h5", "w") as h5_obj:
        h5_obj["my_bool_data"] = boolean_data

    with h5py.File(f"{runpath}/my_data.h5", "r") as h5_obj:
        print(h5_obj["my_bool_data"].shape, h5_obj["my_bool_data"].dtype)

    spec = {"/my_bool_data": tf.TensorSpec(shape=(None,), dtype=tf.bool)}
    h5_tensors = tfio.IOTensor.from_hdf5(f"{runpath}/my_data.h5", spec=spec)

    print("H5 DATA: ", h5_tensors("/my_bool_data").to_tensor())

    assert np.array_equal(boolean_data, h5_tensors("/my_bool_data").to_tensor())

    mapping = {"SOLID": 0, "LIQUID": 1, "GAS": 2, "PLASMA": 3}
    dtype = h5py.special_dtype(enum=(np.int16, mapping))
    enum_data = np.asarray([0, 1, 2, 3])

    with h5py.File(f"{runpath}/my_enum_data.h5", "w") as h5_obj:
        dset = h5_obj.create_dataset("my_enum_data", [4], dtype=dtype)
        dset = enum_data

    with h5py.File(f"{runpath}/my_enum_data.h5", "r") as h5_obj:
        print(h5_obj["my_enum_data"].shape, h5_obj["my_enum_data"].dtype)

    spec = {"/my_enum_data": tf.TensorSpec(shape=(None,), dtype=tf.bool)}
    with pytest.raises(
        tf.errors.InvalidArgumentError, match=r".*unsupported data class for enum.*"
    ):
        h5_tensors = tfio.IOTensor.from_hdf5(f"{runpath}/my_enum_data.h5", spec=spec)

    shutil.rmtree(runpath)


if __name__ == "__main__":
    test.main()
