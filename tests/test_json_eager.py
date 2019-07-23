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
  x_test = [[1.1,2],[2.1,3]]
  y_test = [[2.2,3],[1.2,3]]
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
  feature_dataset = json_io.JSONDataset(feature_filename, columns=feature_list, dtypes=[tf.float64, tf.int64])
  label_dataset = json_io.JSONDataset(label_filename, columns=label_list, dtypes=[tf.float64, tf.int64])

  i = 0
  for record in feature_dataset:
    v_x = x_test[i]
    for index in range(len(record)):
      assert v_x[index] == record[index].numpy()
    i += 1
  assert i == len(y_test)

  ## Test of the reverse order of the columns
  feature_list = ["integerfeature", "floatfeature"]
  feature_dataset = json_io.JSONDataset(feature_filename, columns=feature_list, dtypes=[tf.int64, tf.float64])

  i = 0
  for record in feature_dataset:
    v_x = np.flip(x_test[i])
    for index in range(len(record)):
      assert v_x[index] == record[index].numpy()
    i += 1
  assert i == len(y_test)

  i = 0
  for record in label_dataset:
    v_y = y_test[i]
    for index in range(len(record)):
      assert v_y[index] == record[index].numpy()
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
    for index in range(len(j_x)):
      assert v_x[index] == j_x[index].numpy()
    for index in range(len(j_y)):
      assert v_y[index] == j_y[index].numpy()
    i += 1
  assert i == len(y_test)


if __name__ == "__main__":
  test.main()
