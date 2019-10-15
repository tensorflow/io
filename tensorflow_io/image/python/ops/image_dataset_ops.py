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
"""Image Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import tensorflow as tf
from tensorflow import dtypes
from tensorflow.compat.v1 import data
from tensorflow_io.core.python.ops import core_ops

warnings.warn(
    "Image Dataset (WebP/GIF/TIFF are deprecated and "
    "may be removed in the next release, please use "
    "image ops such as decode_webp for future usage",
    DeprecationWarning)


class WebPDataset(data.Dataset):
  """A WebP Image File Dataset that reads the WebP file."""

  def __init__(self, filenames):
    """Create a `WebPDataset`.

      filenames: A `tf.string` tensor containing one or more filenames.
    """
    self._filenames = tf.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    super(WebPDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return core_ops.io_web_p_dataset(self._filenames)

  @property
  def output_classes(self):
    return tf.Tensor

  @property
  def output_shapes(self):
    return tf.TensorShape([None, None, None])

  @property
  def output_types(self):
    return dtypes.uint8

class TIFFDataset(data.Dataset):
  """A TIFF Image File Dataset that reads the TIFF file."""

  def __init__(self, filenames):
    """Create a `TIFFDataset`.

      filenames: A `tf.string` tensor containing one or more filenames.
    """
    self._filenames = tf.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    super(TIFFDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return core_ops.io_tiff_dataset(self._filenames)

  @property
  def output_classes(self):
    return tf.Tensor

  @property
  def output_shapes(self):
    return tf.TensorShape([None, None, None])

  @property
  def output_types(self):
    return dtypes.uint8

class GIFDataset(data.Dataset):
  """A GIF Image File Dataset that reads the GIF file."""

  def __init__(self, filenames):
    """Create a `GIFDataset`.
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    self._filenames = tf.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    super(GIFDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return core_ops.io_gif_dataset(self._filenames)

  @property
  def output_classes(self):
    return tf.Tensor

  @property
  def output_shapes(self):
    return tf.TensorShape([None, None, None])

  @property
  def output_types(self):
    return dtypes.uint8

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
