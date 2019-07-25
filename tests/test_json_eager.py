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
"""Tests for JSON Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()

import tensorflow_io.json as json_io  # pylint: disable=wrong-import-position

def test_json_dataset():
  """Test case for JSON Dataset.
  """
  x_test = [[1.1, 2], [2.1, 3]]
  y_test = [[2.2, 3], [1.2, 3]]
  feature_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_json",
      "feature.json")
  label_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_json",
      "label.json")

  feature_list = ["floatfeature", "integerfeature"]
  label_list = ["floatlabel", "integerlabel"]
  feature_dataset = json_io.JSONDataset(
      feature_filename,
      feature_list,
      [tf.float64, tf.int64])
  label_dataset = json_io.JSONDataset(
      label_filename,
      label_list,
      [tf.float64, tf.int64])

  i = 0
  for record in feature_dataset:
    v_x = x_test[i]
    for index, val in enumerate(record):
      assert v_x[index] == val.numpy()
    i += 1
  assert i == len(y_test)

  ## Test of the reverse order of the columns
  feature_list = ["integerfeature", "floatfeature"]
  feature_dataset = json_io.JSONDataset(
      feature_filename,
      feature_list,
      [tf.int64, tf.float64])

  i = 0
  for record in feature_dataset:
    v_x = np.flip(x_test[i])
    for index, val in enumerate(record):
      assert v_x[index] == val.numpy()
    i += 1
  assert i == len(y_test)

  i = 0
  for record in label_dataset:
    v_y = y_test[i]
    for index, val in enumerate(record):
      assert v_y[index] == val.numpy()
    i += 1
  assert i == len(y_test)

  dataset = tf.data.Dataset.zip((
      feature_dataset,
      label_dataset
  ))

  i = 0
  for (j_x, j_y) in dataset:
    v_x = np.flip(x_test[i])
    v_y = y_test[i]
    for index, x in enumerate(j_x):
      assert v_x[index] == x.numpy()
    for index, y in enumerate(j_y):
      assert v_y[index] == y.numpy()
    i += 1
  assert i == len(y_test)

def test_json_keras():
  """Test case for JSONDataset with keras."""
  feature_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_json",
      "iris.json")
  label_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_json",
      "species.json")

  feature_list = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']
  label_list = ["species"]
  feature_types = [tf.float64, tf.float64, tf.float64, tf.float64]
  label_types = [tf.int64]
  feature_dataset = json_io.JSONDataset(
      feature_filename,
      feature_list,
      feature_types,
      batch=32)
  label_dataset = json_io.JSONDataset(
      label_filename,
      label_list,
      label_types,
      batch=32)
  dataset = tf.data.Dataset.zip((
      feature_dataset,
      label_dataset
  ))
  def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features), axis=1)
    return features, labels
  dataset = dataset.map(pack_features_vector)

  model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
      tf.keras.layers.Dense(10, activation=tf.nn.relu),
      tf.keras.layers.Dense(3)
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  model.fit(dataset, epochs=5)

if __name__ == "__main__":
  test.main()
