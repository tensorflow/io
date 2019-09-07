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

import time
import threading

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io as tfio # pylint: disable=wrong-import-position
from tensorflow_io.core.python.ops.server_dataset_ops import ServerDataset

def test_io_server():
  """test_io_server"""
  server = tfio.IOServer()
  def client():
    tensor = tfio.IOTensor.from_tensor(tf.constant(["12345", "67890"]))
    tensor.to_server("123")
    return None
  x = threading.Thread(target=client)
  x.start()
  time.sleep(5)
  dataset = ServerDataset(server, "123")
  for d in dataset:
    print("DATA: ", d.numpy())
