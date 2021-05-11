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
"""Test tfio.IOLayer"""

import os
import sys
import time
import tempfile
import pytest
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio

if sys.platform == "darwin":
    pytest.skip("TODO: macOS is failing", allow_module_level=True)


@pytest.fixture(name="fashion_mnist", scope="module")
def fixture_fashion_mnist():
    """fixture_fashion_mnist"""
    classes = [
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

    class MNISTClassNamesLayer(tf.keras.layers.Layer):
        """MNISTClassNamesLayer"""

        def __init__(self):
            self._classes = tf.constant(classes)
            super().__init__(trainable=False)

        def call(self, inputs):  # pylint: disable=arguments-differ
            content = tf.argmax(inputs, axis=1)
            content = tf.gather(self._classes, content)
            return content

    (
        (train_images, train_labels),
        (test_images, _),
    ) = tf.keras.datasets.fashion_mnist.load_data()

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

    predictions = model.predict(test_images)

    predictions = [classes[v] for v in np.argmax(predictions, axis=1)]

    return model, test_images, predictions, MNISTClassNamesLayer()


# Note: IOLayerHelper need to have:
# 1) func(): returns the layer creation function
# 2) check(): returns the check function
class TextIOLayerHelper:
    """TextIOLayerHelper"""

    def func(self):
        f, filename = tempfile.mkstemp()
        os.close(f)
        self._filename = filename
        return tfio.experimental.IOLayer.text(filename)

    def check(self, images, predictions):
        f = tf.data.TextLineDataset(self._filename)
        lines = list(f)
        assert np.all(lines[5] == predictions[5].rstrip())

        assert len(lines) == len(images)


class KafkaIOLayerHelper:
    """KafkaIOLayerHelper"""

    def func(self):
        channel = "e{}e".format(time.time())
        self._topic = "io-layer-test-" + channel
        return tfio.experimental.IOLayer.kafka(self._topic)

    def check(self, images, predictions):
        import tensorflow_io.kafka as kafka_io

        f = kafka_io.KafkaDataset(topics=[self._topic], group="test", eof=True)
        lines = list(f)
        assert np.all(lines == predictions)

        assert len(lines) == len(images)


@pytest.mark.parametrize(
    ("helper"), [(TextIOLayerHelper()), (KafkaIOLayerHelper()),], ids=["text", "kafka"],
)
def test_io_layer(fashion_mnist, helper):
    """test_text_io_layer"""
    model, images, predictions, processing_layer = fashion_mnist

    model.summary()

    io_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=helper.func()(processing_layer(model.layers[-1].output)),
    )

    predictions = io_model.predict(images)

    io_model.layers[-1].sync()

    helper.check(images, predictions)


if __name__ == "__main__":
    test.main()
