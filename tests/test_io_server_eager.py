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
"""Test IOServer"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import threading
import numpy as np

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io as tfio # pylint: disable=wrong-import-position
from tensorflow_io.core.python.ops.server_dataset_ops import ServerDataset # pylint: disable=wrong-import-position
from tensorflow_io.grpc.python.ops import grpc_io_server_client # pylint: disable=wrong-import-position

def test_io_server():
  """test_io_server"""
  mnist_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_mnist",
      "mnist.npz")
  with np.load(mnist_filename) as f:
    (x_test, y_test) = f['x_test'], f['y_test']
  x_test = tf.constant(x_test)
  y_test = tf.constant(y_test)


  # Build a model with normal data, for demo purposes,
  # just use x_test and y_test as x_train and y_train
  x_train, y_train = x_test, y_test
  x_train = tf.image.convert_image_dtype(x_train, tf.float32)
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=5)


  ########################################################
  # component is a key to identify the stream at the server
  component = "1234567"

  server = tfio.IOServer()
  # TODO: y_test could be dropped for inference
  def client():
    client = grpc_io_server_client.GRPCIOServerClient()
    return client.send(component, x_test, y_test)
  x = threading.Thread(target=client)
  x.start()

  # TODO: y_test could be dropped for inference
  time.sleep(5)
  dataset = ServerDataset(
      server, component,
      tf.TensorSpec(x_test.shape, x_test.dtype),
      tf.TensorSpec(y_test.shape, y_test.dtype))
  # dataset is ready to be used
  ########################################################

  # TODO: this line could be dropped for inference
  # if the previous TODOs are in place:
  dataset = dataset.map(
      lambda value, label: value).map(
          lambda value: tf.image.convert_image_dtype(value, tf.float32))

  dataset = dataset.batch(1)
  predict = model.predict(dataset)
  print("PREDICT: ", predict.dtype, predict.shape)
