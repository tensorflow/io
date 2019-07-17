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
"""Tests for Kafka Output Sequence."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pytest
import numpy as np

import tensorflow as tf
import tensorflow_io.kafka as kafka_io

@pytest.mark.skipif(
    not (hasattr(tf, "version") and
         tf.version.VERSION.startswith("2.0.")), reason=None)
def test_kafka_output_sequence():
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
    """KafkaOutputCallback"""
    def __init__(self, batch_size, topic, servers):
      self._sequence = kafka_io.KafkaOutputSequence(
          topic=topic, servers=servers)
      self._batch_size = batch_size
    def on_predict_batch_end(self, batch, logs=None):
      index = batch * self._batch_size
      for outputs in logs['outputs']:
        for output in outputs:
          self._sequence.setitem(index, class_names[np.argmax(output)])
          index += 1
    def flush(self):
      self._sequence.flush()

  channel = "e{}e".format(time.time())
  topic = "test_"+channel

  # By default batch size is 32
  output = OutputCallback(32, topic, "localhost")
  predictions = model.predict(test_images, callbacks=[output])
  output.flush()

  predictions = [class_names[v] for v in np.argmax(predictions, axis=1)]

  # Reading from `test_e(time)e` we should get the same result
  dataset = kafka_io.KafkaDataset(topics=[topic], group="test", eof=True)
  for entry, prediction in zip(dataset, predictions):
    assert entry.numpy() == prediction.encode()
