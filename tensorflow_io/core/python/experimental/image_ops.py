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

import tensorflow as tf

from tensorflow_io.core.python.ops import core_ops


def draw_bounding_boxes(images, boxes, texts=None, colors=None, name=None):
    """
    Draw bounding boxes on a batch of images.

    Args:
      images: A Tensor. Must be one of the following types: float32, half.
        4-D with shape [batch, height, width, depth]. A batch of images.
      boxes: A Tensor of type float32. 3-D with shape
        [batch, num_bounding_boxes, 4] containing bounding boxes.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
    """
    if texts is None:
        texts = []
    if colors is None:
        colors = [[]]
    return core_ops.io_draw_bounding_boxes_v3(images, boxes, colors, texts, name=name)


def decode_jpeg_exif(contents, name=None):
    """
    Decode Exif information from an JPEG image.

    TODO: Add additional fields besides orientation.

    Args:
      contents: A `Tensor` of type `string`. 0-D.  The JPEG-encoded image.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `int64` for orientation.
    """
    return core_ops.io_decode_jpeg_exif(contents, name=name)


def decode_tiff_info(contents, name=None):
    """
    Decode a TIFF-encoded image meta data.

    Args:
      contents: A `Tensor` of type `string`. 0-D.  The TIFF-encoded image.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
    """
    shape, dtype = core_ops.io_decode_tiff_info(contents, name=name)
    return shape, dtype


def decode_tiff(contents, index=0, name=None):
    """
    Decode a TIFF-encoded image to a uint8 tensor.

    Args:
      contents: A `Tensor` of type `string`. 0-D.  The TIFF-encoded image.
      index: A `Tensor` of type int64. 0-D. The 0-based index of the frame
        inside TIFF-encoded image.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
    """
    return core_ops.io_decode_tiff(contents, index, name=name)


def decode_exr_info(contents, name=None):
    """
    Decode a EXR-encoded image meta data.

    Args:
      contents: A `Tensor` of type `string`. 0-D.  The EXR-encoded image.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
    """
    shape, dtype, channel = core_ops.io_decode_exr_info(contents, name=name)
    return shape, dtype, channel


def decode_exr(contents, index, channel, dtype, name=None):
    """
    Decode a EXR-encoded image to a uint8 tensor.

    Args:
      contents: A `Tensor` of type `string`. 0-D.  The EXR-encoded image.
      index: A `Tensor` of type int64. 0-D. The 0-based index of the frame
        inside EXR-encoded image.
      channel: A `Tensor` of type string. 0-D. The channel inside the image.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
    """
    return core_ops.io_decode_exr(
        contents, index=index, channel=channel, dtype=dtype, name=name
    )


def decode_pnm(contents, dtype=tf.uint8, name=None):
    """
    Decode a PNM-encoded image to a uint8 tensor.

    Args:
      contents: A `Tensor` of type `string`. 0-D.  The PNM-encoded image.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
    """
    return core_ops.io_decode_pnm(contents, dtype=dtype, name=name)


def decode_hdr(contents, name=None):
    """
    Decode a HDR-encoded image to a tf.float tensor.

    Args:
      contents: A `Tensor` of type `string`. 0-D.  The HDR-encoded image.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `float` and shape of `[height, width, 3]` (RGB).
    """
    return core_ops.io_decode_hdr(contents, name=name)


def decode_nv12(contents, size, name=None):
    """
    Decode a NV12-encoded image to a uint8 tensor.

    Args:
      contents: A `Tensor` of type `string`. 0-D.  The NV12-encoded image.
      size: A 1-D int32 Tensor of 2 elements: height, width. The size
        for the images.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `uint8` and shape of `[height, width, 3]` (RGB).
    """
    return core_ops.io_decode_nv12(contents, size=size, name=name)


def decode_yuy2(contents, size, name=None):
    """
    Decode a YUY2-encoded image to a uint8 tensor.

    Args:
      contents: A `Tensor` of type `string`. 0-D.  The YUY2-encoded image.
      size: A 1-D int32 Tensor of 2 elements: height, width. The size
        for the images.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `uint8` and shape of `[height, width, 3]` (RGB).
    """
    return core_ops.io_decode_yuy2(contents, size=size, name=name)


def decode_avif(contents, name=None):
    """
    Decode a AVIF-encoded image to a uint8 tensor.

    Args:
      contents: A `Tensor` of type `string`. 0-D.  The AVIF-encoded image.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `uint8` and shape of `[height, width, 3]` (RGB).
    """
    return core_ops.io_decode_avif(contents, name=name)


def decode_jp2(contents, dtype=tf.uint8, name=None):
    """
    Decode a JPEG2000-encoded image to a uint8 tensor.

    Args:
      contents: A `Tensor` of type `string`. 0-D.  The JPEG200-encoded image.
      dtype: Data type of the decoded image. Default `tf.uint8`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `uint8` and shape of `[height, width, 3]` (RGB).
    """
    return core_ops.io_decode_jpeg2k(contents, dtype=dtype, name=name)


def decode_obj(contents, name=None):
    """
    Decode a Wavefront (obj) file into a float32 tensor.

    Args:
      contents: A 0-dimensional Tensor of type string, i.e the
        content of the Wavefront (.obj) file.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `float32` and shape of `[n, 3]` for vertices.
    """
    return core_ops.io_decode_obj(contents, name=name)
