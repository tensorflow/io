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

import sys
import warnings

from tensorflow_io.core.python.ops import _load_library


def _load_libraries(p):
    """load_dependency_and_library"""
    for library in p:
        try:
            v = _load_library(library)
            # Only Linux utilize the library for
            # EncodeAACFunctionFiniFFmpeg
            # EncodeAACFunctionInitFFmpeg
            # EncodeAACFunctionCallFFmpeg
            # DecodeAACFunctionFiniFFmpeg
            # DecodeAACFunctionInitFFmpeg
            # DecodeAACFunctionCallFFmpeg
            l = (
                _load_library(library, "dependency")
                if sys.platform == "linux"
                else None
            )
            if v is not None:
                return v, l
        except NotImplementedError as e:
            warnings.warn("could not load {}: {}".format(library, e))
        NotImplementedError
    raise NotImplementedError("could not find ffmpeg after search through ", p)


_ffmpeg_ops, _decode_ops = _load_libraries(
    [
        "libtensorflow_io_ffmpeg_4.2.so",
        "libtensorflow_io_ffmpeg_3.4.so",
        "libtensorflow_io_ffmpeg_2.8.so",
    ]
)

io_ffmpeg_readable_init = _ffmpeg_ops.io_ffmpeg_readable_init
io_ffmpeg_readable_spec = _ffmpeg_ops.io_ffmpeg_readable_spec
io_ffmpeg_readable_read = _ffmpeg_ops.io_ffmpeg_readable_read
io_ffmpeg_decode_video = _ffmpeg_ops.io_ffmpeg_decode_video
io_ffmpeg_audio_readable_init = _ffmpeg_ops.io_ffmpeg_audio_readable_init
io_ffmpeg_audio_readable_next = _ffmpeg_ops.io_ffmpeg_audio_readable_next
io_ffmpeg_video_readable_init = _ffmpeg_ops.io_ffmpeg_video_readable_init
io_ffmpeg_video_readable_next = _ffmpeg_ops.io_ffmpeg_video_readable_next
