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
import tensorflow as tf
import tensorflow_io.json as json_io


def test_json():
  """test_json"""
  feature_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_json",
      "feature.json")
  label_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_json",
      "label.json")
  d_train_feature = json_io.JSONDataset(
      feature_filename,
  )
  d_train_label = json_io.JSONDataset(
      label_filename,
  )

  d_train = tf.data.Dataset.zip((
      d_train_feature,
      d_train_label
  ))
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(2, input_shape=(1,)),
  ])
  model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

  model.fit(d_train, epochs=5)

if __name__ == "__main__":
  test.main()
