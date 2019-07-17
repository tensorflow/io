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
  json_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_json",
      "json_test.npz")
  with np.load(json_filename) as f:
    (x_test, y_test) = f["x_test"], f["y_test"]
  feature_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_json",
      "feature.json")
  label_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_json",
      "label.json")

  feature_dataset = json_io.JSONDataset(feature_filename)
  label_dataset = json_io.JSONDataset(label_filename)

  i = 0
  for j_x in feature_dataset:
    v_x = x_test[i]
    assert np.alltrue(v_x == j_x.numpy())
    i += 1
  assert i == len(y_test)

  ## Test of the reverse order of the columns
  feature_list = ["integerfeature", "floatfeature"]
  feature_dataset = json_io.JSONDataset(feature_filename, feature_list)

  i = 0
  for j_x in feature_dataset:
    v_x = np.flip(x_test[i])
    assert np.alltrue(v_x == j_x.numpy())
    i += 1
  assert i == len(y_test)

  i = 0
  for j_y in label_dataset:
    v_y = y_test[i]
    assert np.alltrue(v_y == j_y.numpy())
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
    assert np.alltrue(v_y == j_y.numpy())
    assert np.alltrue(v_x == j_x.numpy())
    i += 1
  assert i == len(y_test)


if __name__ == "__main__":
  test.main()
