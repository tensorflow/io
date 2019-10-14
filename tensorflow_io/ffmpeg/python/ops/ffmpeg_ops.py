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

import warnings

from tensorflow_io.core.python.ops import ffmpeg_ops
from tensorflow_io.core.python.ops import ffmpeg_dataset_ops

warnings.warn(
    "The tensorflow_io.ffmpeg.AudioDataset/VideoDataset is "
    "deprecated. Please look for tfio.IODataset.from_ffmpeg "
    "for reading LMDB key/value pairs into tensorflow.",
    DeprecationWarning)

decode_video = ffmpeg_ops.io_ffmpeg_decode_video

class AudioDataset(ffmpeg_dataset_ops.FFmpegIODataset):
  """A Audio File Dataset that reads the audio file."""

  def __init__(self, filename, stream="a:0"):
    super(AudioDataset, self).__init__(filename, stream)

class VideoDataset(ffmpeg_dataset_ops.FFmpegIODataset):
  """A Video File Dataset that reads the video file."""

  def __init__(self, filename, stream="v:0"):
    super(VideoDataset, self).__init__(filename, stream)
