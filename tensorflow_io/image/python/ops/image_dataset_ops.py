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

import os

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
image_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_image_ops.so'))


class WebPDataset(dataset_ops.DatasetSource):
  """A WebP Image File Dataset that reads the WebP file."""

  def __init__(self, filenames):
    """Create a `WebPDataset`.

      filenames: A `tf.string` tensor containing one or more filenames.
    """
    super(WebPDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")

  def _as_variant_tensor(self):
    return image_ops.web_p_dataset( self._filenames)

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return (tensor_shape.TensorShape([None, None, None]))

  @property
  def output_types(self):
    return dtypes.uint8

class TIFFDataset(dataset_ops.DatasetSource):
  """A TIFF Image File Dataset that reads the TIFF file."""

  def __init__(self, filenames):
    """Create a `TIFFDataset`.

      filenames: A `tf.string` tensor containing one or more filenames.
    """
    super(TIFFDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")

  def _as_variant_tensor(self):
    return image_ops.tiff_dataset(self._filenames)

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return (tensor_shape.TensorShape([None, None, None]))

  @property
  def output_types(self):
    return dtypes.uint8
