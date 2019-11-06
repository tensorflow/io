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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
  return core_ops.io_draw_bounding_boxes_v3(
      images, boxes, colors, texts, name=name)

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
      contents, index=index, channel=channel, dtype=dtype, name=name)
