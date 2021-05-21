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
"""FFmpeg"""


def decode_video(content, index=0, name=None):
    """Decode video stream from a video file.

    Args:
      content: A `Tensor` of type `string`.
      index: The stream index.

    Returns:
      value: A `uint8` Tensor.
    """
    from tensorflow_io.core.python.ops import (  # pylint: disable=import-outside-toplevel
        ffmpeg_ops,
    )

    return ffmpeg_ops.io_ffmpeg_decode_video(content, index, name=name)
