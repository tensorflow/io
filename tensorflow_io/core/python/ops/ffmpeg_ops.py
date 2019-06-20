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
"""Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes

from tensorflow_io import _load_library

import _ctypes

def _load_dependency_and_library(p):
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

_ffmpeg_ops = _load_dependency_and_library({
    'libtensorflowio_ffmpeg_3.4.so': [
        "libavformat.so.57",
        "libavformat.so.57",
        "libavutil.so.55",
        "libswscale.so.4",
    ],
    'libtensorflowio_ffmpeg_2.8.so': [
        "libavformat-ffmpeg.so.56",
        "libavcodec-ffmpeg.so.56",
        "libavutil-ffmpeg.so.54",
        "libswscale-ffmpeg.so.3",
    ],
    'libtensorflowio_libav_9.20.so': [
        "libavformat.so.54",
        "libavcodec.so.54",
        "libavutil.so.52",
        "libswscale.so.2",
    ],
})

audio_input = _ffmpeg_ops.audio_input
video_input = _ffmpeg_ops.video_input
audio_dataset = _ffmpeg_ops.audio_dataset
video_dataset = _ffmpeg_ops.video_dataset
