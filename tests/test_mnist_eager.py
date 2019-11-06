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
import sys
import pytest
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio

def test_mnist_dataset():
  """Test case for MNIST Dataset.
  """
  mnist_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_mnist",
      "mnist.npz")
  with np.load(mnist_filename) as f:
    (x_test, y_test) = f['x_test'], f['y_test']

  image_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_mnist",
      "t10k-images-idx3-ubyte.gz")
  label_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_mnist",
      "t10k-labels-idx1-ubyte.gz")

  image_dataset = tfio.IODataset.from_mnist(images=image_filename)
  label_dataset = tfio.IODataset.from_mnist(labels=label_filename)

  i = 0
  for m_x in image_dataset:
    v_x = x_test[i]
    assert np.alltrue(v_x == m_x.numpy())
    i += 1
  assert i == len(y_test)

  i = 0
  for m_y in label_dataset:
    v_y = y_test[i]
    assert np.alltrue(v_y == m_y.numpy())
    i += 1
  assert i == len(y_test)

  dataset = tfio.IODataset.from_mnist(
      image_filename, label_filename)

  i = 0
  for (m_x, m_y) in dataset:
    v_x = x_test[i]
    v_y = y_test[i]
    assert np.alltrue(v_y == m_y.numpy())
    assert np.alltrue(v_x == m_x.numpy())
    i += 1
  assert i == len(y_test)

def test_mnist_tutorial():
  """test_mnist_tutorial"""
  # Note: use http here as we support http file system.
  image_filename = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
  label_filename = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
  d_train = tfio.IODataset.from_mnist(
      image_filename, label_filename).batch(1000)

  d_train = d_train.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y))

  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(d_train, epochs=5)

@pytest.mark.skipif(sys.platform == "darwin", reason=None)
def test_mnist_tutorial_uncompressed():
  """test_mnist_tutorial_uncompressed"""
  image_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_mnist",
      "t10k-images-idx3-ubyte")
  label_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_mnist",
      "t10k-labels-idx1-ubyte")
  d_train = tfio.IODataset.from_mnist(
      image_filename, label_filename).batch(1)

  d_train = d_train.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y))

  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(d_train, epochs=5)

if __name__ == "__main__":
  test.main()
