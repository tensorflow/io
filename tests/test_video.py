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
if sys.platform == "darwin":
  pytest.skip("video is not supported on macOS yet", allow_module_level=True)
import tensorflow_io.video as video_io  # pylint: disable=wrong-import-position

video_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_video", "small.mp4")

def test_video_predict():
  model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
  x = video_io.VideoDataset(video_path, batch=1).map(lambda x: tf.keras.applications.resnet50.preprocess_input(tf.image.resize(x, (224, 224))))
  y = model.predict(x)
  p = tf.keras.applications.resnet50.decode_predictions(y, top=1)
  assert len(p) == 166
