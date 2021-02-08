"""Tests for the super_serial.py serialization module."""
import os
import tempfile

import numpy as np
import tensorflow as tf

import tensorflow_io.core.python.api.v0.super_serial as ser


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
    ser.save(ds, tfrecord_path=tfrecord_path, header_path=header_path)
    new_ds = ser.load(tfrecord_path=tfrecord_path, header_path=header_path)

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
