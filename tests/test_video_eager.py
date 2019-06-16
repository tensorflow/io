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
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
if sys.platform == "darwin":
  pytest.skip("video is not supported on macOS yet", allow_module_level=True)
import tensorflow_io.video as video_io  # pylint: disable=wrong-import-position

video_path = "file://" + os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_video", "small.mp4")
def test_video_dataset():
  """test_video_dataset"""
  num_repeats = 2

  video_dataset = video_io.VideoDataset([video_path]).repeat(num_repeats)

  i = 0
  for v in video_dataset:
    assert v.shape == (320, 560, 3)
    i += 1
  assert i == 166 * num_repeats
