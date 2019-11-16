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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tempfile
import pytest
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_io.kafka as kafka_io

@pytest.fixture(name="fashion_mnist", scope="module")
def fixture_fashion_mnist():
  """fixture_fashion_mnist"""
  classes = [
      'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  class MNISTClassNamesLayer(tf.keras.layers.Layer):
    """MNISTClassNamesLayer"""
    def __init__(self):
      self._classes = tf.constant(classes)
      super(MNISTClassNamesLayer, self).__init__(trainable=False)

    def call(self, inputs):
      content = tf.argmax(inputs, axis=1)
      content = tf.gather(self._classes, content)
      return content

  ((train_images, train_labels),
   (test_images, _)) = tf.keras.datasets.fashion_mnist.load_data()

  train_images = train_images / 255.0
  test_images = test_images / 255.0

  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(train_images, train_labels, epochs=5)

  predictions = model.predict(test_images)

  predictions = [classes[v] for v in np.argmax(predictions, axis=1)]

  return model, test_images, predictions, MNISTClassNamesLayer()

def test_text_io_layer(fashion_mnist):
  """test_text_io_layer"""
  model, images, predictions, processing_layer = fashion_mnist

  model.summary()

  f, filename = tempfile.mkstemp()
  os.close(f)

  io_model = tf.keras.models.Model(
      inputs=model.input,
      outputs=tfio.IOLayer.text(filename)(
          processing_layer(model.layers[-1].output)))

  predictions = io_model.predict(images)

  io_model.layers[-1].sync()

  f = tf.data.TextLineDataset(filename)
  lines = [line for line in f]
  assert np.all(lines == predictions)

  assert len(lines) == len(images)

def test_kafka_io_layer(fashion_mnist):
  """test_kafka_io_layer"""
  model, images, predictions, processing_layer = fashion_mnist

  model.summary()

  # Reading from `test_e(time)e` we should get the same result
  channel = "e{}e".format(time.time())
  topic = "io-layer-test-"+channel

  io_model = tf.keras.models.Model(
      inputs=model.input,
      outputs=tfio.IOLayer.kafka(topic)(
          processing_layer(model.layers[-1].output)))

  predictions = io_model.predict(images)

  io_model.layers[-1].sync()

  f = kafka_io.KafkaDataset(topics=[topic], group="test", eof=True)
  lines = [line for line in f]
  assert np.all(lines == predictions)

  assert len(lines) == len(images)

if __name__ == "__main__":
  test.main()
