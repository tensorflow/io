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
"""Test IOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io as tfio # pylint: disable=wrong-import-position

def test_window():
  """test_window"""
  value = [[e] for e in range(100)]
  value = tfio.IOTensor.from_tensor(tf.constant(value))
  value = value.window(3)
  expected_value = [[e, e+1, e+2] for e in range(98)]
  assert np.all(value.to_tensor().numpy() == expected_value)

  v = tfio.IOTensor.from_tensor(tf.constant([1, 2, 3, 4, 5]))
  v = v.window(3)
  assert np.all(v.to_tensor().numpy() == [[1, 2, 3], [2, 3, 4], [3, 4, 5]])

def test_window_to_dataset():
  """test_window_to_dataset"""
  value = [[e] for e in range(100)]
  value = tfio.IOTensor.from_tensor(tf.constant(value))
  value = value.window(3)
  expected_value = [[e, e+1, e+2] for e in range(98)]
  dataset = value.to_dataset()
  dataset_value = [d.numpy().tolist() for d in dataset]
  assert np.all(dataset_value == expected_value)
