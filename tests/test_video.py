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
"""test_video.py"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow import errors # pylint: disable=wrong-import-position
from tensorflow import image  # pylint: disable=wrong-import-position
from tensorflow.keras.applications.resnet50 import ResNet50 # pylint: disable=wrong-import-position
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions # pylint: disable=wrong-import-position

if sys.platform == "darwin":
  pytest.skip("video is not supported on macOS yet", allow_module_level=True)

import tensorflow_io.video as video_io  # pylint: disable=wrong-import-position

video_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_video", "small.mp4")

@pytest.mark.skipif(
    not (hasattr(tf, "version") and
         tf.version.VERSION.startswith("2.0.")), reason=None)
def test_video_predict():
  model = ResNet50(weights='imagenet')
  x = video_io.VideoDataset(video_path).batch(1).map(lambda x: preprocess_input(image.resize(x, (224, 224))))
  y = model.predict(x)
  p = decode_predictions(y, top=1)
  assert len(p) == 166

def test_video_dataset():
  """test_video_dataset"""
  num_repeats = 2

  dataset = video_io.VideoDataset([video_path]).repeat(num_repeats)
  iterator = dataset.make_initializable_iterator()
  init_op = iterator.initializer
  get_next = iterator.get_next()

  with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    for _ in range(num_repeats):
      for _ in range(166):
        v = sess.run(get_next)
        assert v.shape == (320, 560, 3)
    with pytest.raises(errors.OutOfRangeError):
      sess.run(get_next)
