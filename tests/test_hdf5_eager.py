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

import tensorflow as tf
import tensorflow_io as tfio


def test_hdf5():
    """test_hdf5: GitHub issue 841"""

    def create_datasets(runpath, cnt=10):
        os.makedirs(runpath, exist_ok=True)
        for i in range(cnt):
            f = h5py.File("{}/file_{}.h5".format(runpath, i), "w")
            total_samples = np.random.randint(50000, 100000)
            f.create_dataset("features", data=np.random.random((total_samples, 60)))
            f.create_dataset("targets", data=np.random.random((total_samples, 3)))
            f.close()

    runpath = tempfile.mkdtemp()
    create_datasets(runpath)

    for i in range(2):
        cnt = 0
        for p in glob.glob("{}/*.h5".format(runpath)):
            try:
                features = tfio.IODataset.from_hdf5(p, "/features")
                targets = tfio.IODataset.from_hdf5(p, "/targets")
                dataset = tf.data.Dataset.zip((features, targets))

                for t in dataset:
                    cnt += t[0].shape[0]

            except Exception as e:
                print("Failed going through {}".format(p))
                raise e
            print("Success going through {}".format(p))

    print("Iterated {} items".format(cnt))

    shutil.rmtree(runpath)


if __name__ == "__main__":
    test.main()
