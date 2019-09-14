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
from tensorflow_io.core.python.ops.server_dataset_ops import ServerDataset
from tensorflow_io.grpc.python.ops import grpc_io_server_client

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

  ########################################################
  # component is a key to identify the stream at the server
  component = "1234567"

  server = tfio.IOServer()

  def client():
    client = grpc_io_server_client.GRPCIOServerClient()
    return client.send(component, x_test, y_test)
  x = threading.Thread(target=client)
  x.start()

  time.sleep(5)
  dataset = ServerDataset(
      server, component,
      tf.TensorSpec(x_test.shape, x_test.dtype),
      tf.TensorSpec(y_test.shape, y_test.dtype))
  # dataset is ready to be used
  ########################################################

  i = 0
  for d in dataset:
    x, y = d
    assert np.all(x == x_test[i])
    assert np.all(y == y_test[i])
    i += 1
