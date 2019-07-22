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
"""Tests for Image Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io.image as image_io # pylint: disable=wrong-import-position


def test_webp_file_dataset():
  """Test case for WebPDataset.
  """
  filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_image", "sample.webp")

  num_repeats = 2

  dataset = image_io.WebPDataset([filename, filename])
  # Repeat 2 times (2 * 2 = 4 images)
  dataset = dataset.repeat(num_repeats)
  # Drop alpha channel
  dataset = dataset.map(lambda x: x[:, :, :3])
  # Resize to 224 * 224
  dataset = dataset.map(lambda x: tf.keras.applications.resnet50.preprocess_input(tf.image.resize(x, (224, 224))))
  # Batch to 3, still have 4 images (3 + 1)
  dataset = dataset.batch(1)
  model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
  y = model.predict(dataset)
  p = tf.keras.applications.resnet50.decode_predictions(y, top=1)
  for i in p:
    assert i[0][1] == 'pineapple' # not truly a pineapple, though
  assert len(p) == 4

if __name__ == "__main__":
  test.main()
