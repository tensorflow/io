# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the super_serial.py serialization module."""
import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

import tensorflow_io as tfio


def test_serialization():
    """Test super serial saving and loading.
    NOTE- test will only work in eager mode due to list() dataset cast."""
    savefolder = tempfile.TemporaryDirectory()
    savepath = os.path.join(savefolder.name, "temp_dataset")
    tfrecord_path = savepath + ".tfrecord"
    header_path = savepath + ".header"

    # Data
    x = np.linspace(1, 3000, num=3000).reshape(10, 10, 10, 3)
    y = np.linspace(1, 10, num=10).astype(int)
    ds = tf.data.Dataset.from_tensor_slices({"image": x, "label": y})

    # Run
    tfio.experimental.serialization.save_dataset(
        ds, tfrecord_path=tfrecord_path, header_path=header_path
    )
    new_ds = tfio.experimental.serialization.load_dataset(
        tfrecord_path=tfrecord_path, header_path=header_path
    )

    # Test that values were saved and restored
    assert (
        list(ds)[0]["image"].numpy()[0, 0, 0]
        == list(new_ds)[0]["image"].numpy()[0, 0, 0]
    )
    assert list(ds)[0]["label"] == list(new_ds)[0]["label"]

    assert (
        list(ds)[-1]["image"].numpy()[0, 0, 0]
        == list(new_ds)[-1]["image"].numpy()[0, 0, 0]
    )
    assert list(ds)[-1]["label"] == list(new_ds)[-1]["label"]

    # Clean up- folder will disappear on crash as well.
    savefolder.cleanup()


@tf.function
def graph_save_fail():
    """Serial ops is expected to raise an exception when
    trying to save in graph mode."""
    savefolder = tempfile.TemporaryDirectory()
    savepath = os.path.join(savefolder.name, "temp_dataset")
    tfrecord_path = savepath + ".tfrecord"
    header_path = savepath + ".header"

    # Data
    x = np.linspace(1, 3000, num=3000).reshape(10, 10, 10, 3)
    y = np.linspace(1, 10, num=10).astype(int)
    ds = tf.data.Dataset.from_tensor_slices({"image": x, "label": y})

    # Run
    assert os.path.isdir(savefolder.name)
    assert not tf.executing_eagerly()
    tfio.experimental.serialization.save_dataset(
        ds, tfrecord_path=tfrecord_path, header_path=header_path
    )


def test_ensure_graph_fail():
    """Test that super_serial fails in graph mode."""
    with pytest.raises(ValueError):
        graph_save_fail()
