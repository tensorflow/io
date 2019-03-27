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
"""Tests for MNIST Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import pytest
import numpy as np

import tensorflow as tf
import tensorflow_io.text as text_io

@pytest.mark.skipif(
    not (hasattr(tf, "version") and
         tf.version.VERSION.startswith("2.0.")), reason=None)
def test_text_output_sequence():
  """Test case based on fashion mnist tutorial"""
  fashion_mnist = tf.keras.datasets.fashion_mnist
  ((train_images, train_labels),
   (test_images, _)) = fashion_mnist.load_data()

  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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

  class OutputCallback(tf.keras.callbacks.Callback):
    def __init__(self, filename, batch_size):
      self._sequence = text_io.TextOutputSequence(filename)
      self._batch_size = batch_size
    def on_predict_batch_end(self, batch, logs=None):
      index = batch * self._batch_size
      for outputs in logs['outputs']:
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
