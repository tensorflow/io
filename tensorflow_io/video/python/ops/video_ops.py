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
"""Video Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes

import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops as data_ops
from tensorflow_io import _load_library

import _ctypes

def load_dependency_and_library(p):
  """load_dependency_and_library"""
  for library in p:
    # First try load all dependencies with RTLD_LOCAL
    entries = []
    for dependency in p[library]:
      try:
        entries.append(ctypes.CDLL(dependency))
      except OSError:
        pass
    if len(entries) == len(p[library]):
      # Dependencies has been satisfied, load dependencies again with RTLD_GLOBAL, no error is expected
      for dependency in p[library]:
        ctypes.CDLL(dependency, mode=ctypes.RTLD_GLOBAL)
      # Load video_op
      return _load_library(library)
    # Otherwise we dlclose and retry
    entries.reverse()
    for entry in entries:
      _ctypes.dlclose(entry._handle) # pylint: disable=protected-access
  raise NotImplementedError("could not find ffmpeg after search through ", p)

video_ops = load_dependency_and_library({
    '_video_ops_ffmpeg_3.4.so': [
        "libavformat.so.57",
        "libavformat.so.57",
        "libavutil.so.55",
        "libswscale.so.4",
    ],
    '_video_ops_ffmpeg_2.8.so': [
        "libavformat-ffmpeg.so.56",
        "libavcodec-ffmpeg.so.56",
        "libavutil-ffmpeg.so.54",
        "libswscale-ffmpeg.so.3",
    ],
    '_video_ops_libav_9.20.so': [
        "libavformat.so.54",
        "libavcodec.so.54",
        "libavutil.so.52",
        "libswscale.so.2",
    ],
})

class VideoDataset(data_ops.BaseDataset):
  """A Video File Dataset that reads the video file."""

  def __init__(self, filename):
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
    """
    batch = None # TODO: Add batch support
    self._batch = 0 if batch is None else batch
    self._dtypes = [tf.uint8]
    self._shapes = [tf.TensorShape([None, None, 3])]
    super(VideoDataset, self).__init__(
        video_ops.video_dataset(filename),
        self._batch, self._dtypes, self._shapes)
