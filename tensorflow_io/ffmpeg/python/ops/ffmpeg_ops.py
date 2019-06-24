# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FFmpeg Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops as data_ops
from tensorflow_io.core.python.ops import ffmpeg_ops as ffmpeg_ops

class VideoDataset(data_ops.Dataset):
  """A Video File Dataset that reads the video file."""

  def __init__(self, filename, batch=None):
    """Create a `VideoDataset`.

    `VideoDataset` allows a user to read data from a video file with
    ffmpeg. The output of VideoDataset is a sequence of (height, weight, 3)
    tensor in rgb24 format.

    For example:

    ```python
    dataset = VideoDataset("/foo/bar.mp4")
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    while True:
      try:
        print(sess.run(next_element))
      except tf.errors.OutOfRangeError:
        break
    ```

    Args:
      filename: A `tf.string` tensor containing one or more filenames.
      batch: An integer representing the number of consecutive image frames
        to combine in a single batch. If `batch == 0` then each element
        of the dataset has one standalone image frame.
    """
    batch = 0 if batch is None else batch
    dtypes = [tf.uint8]
    shapes = [
        tf.TensorShape([None, None, 3])] if batch == 0 else [
            tf.TensorShape([None, None, None, 3])]
    super(VideoDataset, self).__init__(
        ffmpeg_ops.video_dataset,
        ffmpeg_ops.video_input(filename), batch, dtypes, shapes)

class AudioDataset(data_ops.Dataset):
  """A Audio File Dataset that reads the audio file."""

  def __init__(self, filename, batch=None):
    """Create a `AudioDataset`.
    Args:
      filename: A `tf.string` tensor containing one or more filenames.
    """
    batch = 0 if batch is None else batch
    dtypes = [tf.int16]
    shapes = [
        tf.TensorShape([None])] if batch == 0 else [
            tf.TensorShape([None, None])]
    super(AudioDataset, self).__init__(
        ffmpeg_ops.audio_dataset,
        ffmpeg_ops.audio_input(filename),
        batch, dtypes, shapes)
