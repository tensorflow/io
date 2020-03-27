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
"""Tests for Text Input."""


import os
import re
import tempfile
import pytest
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio


def test_read_text():
    """test_read_text"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_text", "lorem.txt"
    )
    with open(filename, "rb") as f:
        lines = list(f)
    filename = "file://" + filename

    filesize = tf.io.gfile.GFile(filename).size()

    offset = 0
    offsets = []
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    lines = list(zip(offsets, lines))

    for offset, length in [(0, -1), (1, -1), (1000, -1), (100, 1000), (1000, 10000)]:
        entries = tfio.experimental.text.read_text(
            filename, offset=offset, length=length
        )
        if length < 0:
            length = filesize - offset
        expected = [line for (k, line) in lines if offset <= k < offset + length]
        assert entries.shape == len(expected)
        for k, v in enumerate(expected):
            assert entries[k].numpy().decode() + "\n" == v.decode()


@pytest.mark.xfail(reason="TODO")
def test_text_output_sequence():
    """Test case based on fashion mnist tutorial"""
    fashion_mnist = tf.keras.datasets.fashion_mnist
    ((train_images, train_labels), (test_images, _)) = fashion_mnist.load_data()

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(train_images, train_labels, epochs=5)

    class OutputCallback(tf.keras.callbacks.Callback):
        """OutputCallback"""

        def __init__(
            self, filename, batch_size
        ):  # pylint: disable=super-init-not-called
            self._sequence = tfio.experimental.text.TextOutputSequence(filename)
            self._batch_size = batch_size

        def on_predict_batch_end(self, batch, logs=None):
            index = batch * self._batch_size
            for outputs in logs["outputs"]:
                for output in outputs:
                    self._sequence.setitem(index, class_names[np.argmax(output)])
                    index += 1

    f, filename = tempfile.mkstemp()
    os.close(f)
    # By default batch size is 32
    output = OutputCallback(filename, 32)
    predictions = model.predict(test_images, callbacks=[output])
    with open(filename) as f:
        lines = [line.strip() for line in f]
    predictions = [class_names[v] for v in np.argmax(predictions, axis=1)]
    assert len(lines) == len(predictions)
    for line, prediction in zip(lines, predictions):
        assert line == prediction


def test_re2_extract():
    """test_text_input
  """
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_text", "lorem.txt"
    )
    with open(filename, "rb") as f:
        lines = [line.strip() for line in f]
    filename = "file://" + filename

    dataset = tf.data.TextLineDataset(filename).map(
        lambda x: tfio.experimental.text.re2_full_match(x, ".+(ipsum).+(dolor).+")
    )
    i = 0
    for v in dataset:
        r, g = v
        if re.match(b".+(ipsum).+(dolor).+", lines[i]):
            assert r.numpy()
            assert g[0].numpy().decode() == "ipsum"
            assert g[1].numpy().decode() == "dolor"
        else:
            assert not r.numpy()
            assert g[0].numpy().decode() == ""
            assert g[1].numpy().decode() == ""
        i += 1
    assert i == len(lines)


if __name__ == "__main__":
    test.main()
