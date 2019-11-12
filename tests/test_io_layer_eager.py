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
import re
import tempfile
import pytest
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio

@pytest.fixture(name="fashion_mnist", scope="module")
def fixture_fashion_mnist():
  """fixture_fashion_mnist"""
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  ((train_images, train_labels),
   (test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()

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

  return model, test_images, class_names

def test_text_io_layer(fashion_mnist):
  """test_text_io_layer"""
  model, images, classes = fashion_mnist

  model.summary()

  f, filename = tempfile.mkstemp()
  os.close(f)

  io_layer = tfio.IOLayer.text(filename)

  io_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=io_layer(model.layers[-1].output))

  predictions = io_model.predict(images)

  io_layer.sync()

  with open(filename) as f:
    lines = [line.strip() for line in f]
  predictions = [classes[v] for v in np.argmax(predictions, axis=1)]
  assert len(lines) == len(predictions)

if __name__ == "__main__":
  test.main()
