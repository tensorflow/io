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
"""Tests for ImageIOTensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io as tfio # pylint: disable=wrong-import-position


def test_tiff_io_tensor():
  """Test case for TIFFImageIOTensor"""
  width = 560
  height = 320
  channels = 4

  images = []
  for filename in [
      "small-00.png",
      "small-01.png",
      "small-02.png",
      "small-03.png",
      "small-04.png"]:
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "test_image",
                     filename), 'rb') as f:
      png_contents = f.read()
    image_v = tf.image.decode_png(png_contents, channels=channels)
    assert image_v.shape == [height, width, channels]
    images.append(image_v)

  filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_image", "small.tiff")
  filename = "file://" + filename

  tiff = tfio.IOTensor.from_tiff(filename)
  assert tiff.keys == list(range(5))
  for i in tiff.keys:
    assert np.all(images[i].numpy() == tiff(i).to_tensor().numpy())


if __name__ == "__main__":
  test.main()
