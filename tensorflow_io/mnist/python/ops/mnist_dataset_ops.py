# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""MNIST Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
from tensorflow.compat.v2 import data
from tensorflow_io import _load_library
mnist_ops = _load_library('_mnist_ops.so')

class InputDataset(data.Dataset):
  """An InputDataset"""

  def __init__(self, fn, data_input, output_types, output_shapes):
    """Create an InputDataset."""
    self._data_input = data_input
    self._output_types = output_types
    self._output_shapes = output_shapes
    super(InputDataset, self).__init__(fn(
        self._data_input,
        output_types=self._output_types,
        output_shapes=self._output_shapes))

  def _inputs(self):
    return []

  @property
  def _element_structure(self):
    e = [
        tensorflow.data.experimental.TensorStructure(
            p, q.as_list()) for (p, q) in zip(
                self.output_types, self.output_shapes)
    ]
    if len(e) == 1:
      return e[0]
    return tensorflow.data.experimental.NestedStructure(e)

  @property
  def output_types(self):
    return self._output_types

  @property
  def output_shapes(self):
    return self._output_shapes

class MNISTLabelDataset(InputDataset):
  """A MNISTLabelDataset
  """

  def __init__(self, filename):
    """Create a MNISTLabelDataset.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    super(MNISTLabelDataset, self).__init__(
        mnist_ops.mnist_label_dataset,
        mnist_ops.mnist_label_input(filename, ["none", "gz"]),
        [tensorflow.uint8],
        [tensorflow.TensorShape([])]
    )

class MNISTImageDataset(InputDataset):
  """A MNISTImageDataset
  """

  def __init__(self, filename):
    """Create a MNISTImageDataset.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    super(MNISTImageDataset, self).__init__(
        mnist_ops.mnist_image_dataset,
        mnist_ops.mnist_image_input(filename, ["none", "gz"]),
        [tensorflow.uint8],
        [tensorflow.TensorShape([None, None])]
    )

def MNISTDataset(image_filename, label_filename):
  return data.Dataset.zip((
      MNISTImageDataset(image_filename),
      MNISTLabelDataset(label_filename)))
