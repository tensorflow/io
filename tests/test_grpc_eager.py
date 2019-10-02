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
"""Tests for GRPC Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

import tensorflow as tf
import tensorflow_io.grpc as grpc_io

@pytest.mark.skipif(
    not (hasattr(tf, "version") and
         tf.version.VERSION.startswith("2.0.")), reason=None)
def _test_grpc_with_mnist_tutorial():
  """test_mnist_tutorial"""
  (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
  x = grpc_io.GRPCDataset.from_numpy(x_train, batch=1000)
  y = grpc_io.GRPCDataset.from_numpy(y_train, batch=1000)
  for (i, v) in zip(range(0, 50000, 1000), x):
    assert np.alltrue(x_train[i:i+1000, :] == v.numpy())
  for (i, v) in zip(range(0, 50000, 1000), y):
    assert np.alltrue(y_train[i:i+1000] == v.numpy())
  d_train = tf.data.Dataset.zip((x, y))

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
