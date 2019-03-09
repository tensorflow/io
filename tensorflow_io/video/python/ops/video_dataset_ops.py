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
import _ctypes

import tensorflow
from tensorflow import dtypes
from tensorflow.compat.v1 import data
from tensorflow_io import _load_library

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

class VideoDataset(data.Dataset):
  """A Video File Dataset that reads the video file."""

  def __init__(self, filenames):
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
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    self._filenames = tensorflow.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    super(VideoDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return video_ops.video_dataset(self._filenames)

  @property
  def output_classes(self):
    return tensorflow.Tensor

  @property
  def output_shapes(self):
    return tensorflow.TensorShape([None, None, 3])

  @property
  def output_types(self):
    return dtypes.uint8
