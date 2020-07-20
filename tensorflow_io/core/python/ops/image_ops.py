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
"""Image Ops."""

from tensorflow_io.core.python.ops import core_ops


def decode_webp(contents, name=None):
    """
    Decode a WebP-encoded image to a uint8 tensor.

    Args:
      contents: A `Tensor` of type `string`. 0-D.  The WebP-encoded image.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
    """
    return core_ops.io_decode_web_p(contents, name=name)


def encode_bmp(image, name=None):
    """
    Encode a uint8 tensor to bmp image.

    Args:
      image: A Tensor. 3-D uint8 with shape [height, width, channels].
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `string`.
    """
    return core_ops.io_encode_bmp(image, name=name)


def encode_gif(image, name=None):
    """
    Encode a uint8 tensor to gif image.

    Args:
      image: A Tensor. 3-D uint8 with shape [N, H, W, C].
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `string`.
    """
    return core_ops.io_encode_gif(image, name=name)
