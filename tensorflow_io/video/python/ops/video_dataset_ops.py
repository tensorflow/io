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

import os

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
video_ops = None
for f in ['_video_ops_ffmpeg_3.4.so', '_video_ops_ffmpeg_2.8.so', '_video_ops_libav_9.20.so']:
  try:
    video_ops = load_library.load_op_library(resource_loader.get_path_to_datafile(f))
    break;
  except errors_impl.NotFoundError as e:
    print(e)


class VideoDataset(dataset_ops.DatasetSource):
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
    super(VideoDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")

  def _as_variant_tensor(self):
    return video_ops.video_dataset(self._filenames)

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return (tensor_shape.TensorShape([None, None, 3]))

  @property
  def output_types(self):
    return dtypes.uint8
