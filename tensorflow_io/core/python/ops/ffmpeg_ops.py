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

import ctypes
import _ctypes

from tensorflow_io.core.python.ops import _load_library


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
            v = _load_library(library)
            l = _load_library(library, "dependency")
            return v, l
        # Otherwise we dlclose and retry
        entries.reverse()
        for entry in entries:
            _ctypes.dlclose(entry._handle)  # pylint: disable=protected-access
    raise NotImplementedError("could not find ffmpeg after search through ", p)


_ffmpeg_ops, _decode_ops = _load_dependency_and_library(
    {
        "libtensorflow_io_ffmpeg_3.4.so": [
            "libavformat.so.57",
            "libavformat.so.57",
            "libavutil.so.55",
            "libswscale.so.4",
        ],
        "libtensorflow_io_ffmpeg_2.8.so": [
            "libavformat-ffmpeg.so.56",
            "libavcodec-ffmpeg.so.56",
            "libavutil-ffmpeg.so.54",
            "libswscale-ffmpeg.so.3",
        ],
    }
)

io_ffmpeg_readable_init = _ffmpeg_ops.io_ffmpeg_readable_init
io_ffmpeg_readable_spec = _ffmpeg_ops.io_ffmpeg_readable_spec
io_ffmpeg_readable_read = _ffmpeg_ops.io_ffmpeg_readable_read
io_ffmpeg_decode_video = _ffmpeg_ops.io_ffmpeg_decode_video
io_ffmpeg_audio_readable_init = _ffmpeg_ops.io_ffmpeg_audio_readable_init
io_ffmpeg_audio_readable_next = _ffmpeg_ops.io_ffmpeg_audio_readable_next
io_ffmpeg_video_readable_init = _ffmpeg_ops.io_ffmpeg_video_readable_init
io_ffmpeg_video_readable_next = _ffmpeg_ops.io_ffmpeg_video_readable_next
