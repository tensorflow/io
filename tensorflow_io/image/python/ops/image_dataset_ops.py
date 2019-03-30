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

import tensorflow
from tensorflow import dtypes
from tensorflow.compat.v1 import data
from tensorflow_io import _load_library
image_ops = _load_library('_image_ops.so')


class WebPDataset(data.Dataset):
  """A WebP Image File Dataset that reads the WebP file."""

  def __init__(self, filenames):
    """Create a `WebPDataset`.

      filenames: A `tf.string` tensor containing one or more filenames.
    """
    self._filenames = tensorflow.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    super(WebPDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return image_ops.web_p_dataset(self._filenames)

  @property
  def output_classes(self):
    return tensorflow.Tensor

  @property
  def output_shapes(self):
    return tensorflow.TensorShape([None, None, None])

  @property
  def output_types(self):
    return dtypes.uint8

class TIFFDataset(data.Dataset):
  """A TIFF Image File Dataset that reads the TIFF file."""

  def __init__(self, filenames):
    """Create a `TIFFDataset`.

      filenames: A `tf.string` tensor containing one or more filenames.
    """
    self._filenames = tensorflow.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    super(TIFFDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return image_ops.tiff_dataset(self._filenames)

  @property
  def output_classes(self):
    return tensorflow.Tensor

  @property
  def output_shapes(self):
    return tensorflow.TensorShape([None, None, None])

  @property
  def output_types(self):
    return dtypes.uint8

class GIFDataset(data.Dataset):
  """A GIF Image File Dataset that reads the GIF file."""

  def __init__(self, filenames):
    """Create a `GIFDataset`.
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    self._filenames = tensorflow.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    super(GIFDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return image_ops.gif_dataset(self._filenames)

  @property
  def output_classes(self):
    return tensorflow.Tensor

  @property
  def output_shapes(self):
    return tensorflow.TensorShape([None, None, None])

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
  return image_ops.decode_web_p(contents, name=name)
