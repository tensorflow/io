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
"""CIFAR Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
from tensorflow import dtypes
from tensorflow.compat.v1 import data
from tensorflow_io import _load_library
cifar_ops = _load_library('_cifar_ops.so')

class _CIFAR10Dataset(data.Dataset):
  """A CIFAR File Dataset that reads the cifar file."""

  def __init__(self, filename):
    """Create a `CIFARDataset`.

    Args:
      filename: A `tf.string` tensor containing one or more filenames.
    """
    self._data_input = cifar_ops.cifar10_input(filename)
    super(_CIFAR10Dataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return cifar_ops.cifar10_dataset(
        self._data_input,
        output_types=self.output_types,
        output_shapes=self.output_shapes)

  @property
  def output_classes(self):
    return tensorflow.Tensor, tensorflow.Tensor

  @property
  def output_shapes(self):
    return (tensorflow.TensorShape([]), tensorflow.TensorShape([3, 32, 32]))

  @property
  def output_types(self):
    return dtypes.uint8, dtypes.uint8

class CIFAR10Dataset(_CIFAR10Dataset):
  """A CIFAR File Dataset that reads the cifar file."""

  def __init__(self, filename, test=False):
    """Create a `CIFAR10Dataset`.

    Args:
      filename: A `tf.string` tensor containing one or more filenames.
      test: A boolean to indicate if the data input is for test or for train.
    """
    if test:
      self._filename = filename + "|test_batch.bin"
    else:
      entries = ["|data_batch_" + str(e) + ".bin" for e in range(1, 6)]
      self._filename = [filename + entry for entry in entries]
    super(CIFAR10Dataset, self).__init__(self._filename)

  def _as_variant_tensor(self):
    return _CIFAR10Dataset(self._filename).map( # pylint: disable=protected-access
        lambda label, image: (tensorflow.transpose(image, [1, 2, 0]), label))._as_variant_tensor()

  @property
  def output_shapes(self):
    return (tensorflow.TensorShape([32, 32, 3]), tensorflow.TensorShape([]))

class _CIFAR100Dataset(data.Dataset):
  """A CIFAR File Dataset that reads the cifar file."""

  def __init__(self, filename):
    """Create a `CIFAR100Dataset`.

    Args:
      filename: A `tf.string` tensor containing one or more filenames.
    """
    self._data_input = cifar_ops.cifar100_input(filename)
    super(_CIFAR100Dataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return cifar_ops.cifar100_dataset(
        self._data_input,
        output_types=self.output_types,
        output_shapes=self.output_shapes)

  @property
  def output_classes(self):
    return tensorflow.Tensor, tensorflow.Tensor, tensorflow.Tensor

  @property
  def output_shapes(self):
    return (tensorflow.TensorShape([]),
            tensorflow.TensorShape([]),
            tensorflow.TensorShape([3, 32, 32]))

  @property
  def output_types(self):
    return dtypes.uint8, dtypes.uint8, dtypes.uint8

class CIFAR100Dataset(_CIFAR100Dataset):
  """A CIFAR File Dataset that reads the cifar file."""

  def __init__(self, filename, test=False, mode='fine'):
    """Create a `CIFAR100Dataset`.

    Args:
      filename: A `tf.string` tensor containing one or more filenames.
      test: A boolean to indicate if the data input is for test or for train.
      mode: A string indicate if `coarse` or `fine` label is used.
    """
    if test:
      self._filename = filename + "|test.bin"
    else:
      self._filename = filename + "|train.bin"
    self._mode = mode
    super(CIFAR100Dataset, self).__init__(self._filename)

  def _as_variant_tensor(self):
    return _CIFAR100Dataset(self._filename).map( # pylint: disable=protected-access
        lambda coarse, fine, image: (tensorflow.transpose(image, [1, 2, 0]), fine if self._mode == 'fine' else coarse))._as_variant_tensor()

  @property
  def output_classes(self):
    return tensorflow.Tensor, tensorflow.Tensor

  @property
  def output_shapes(self):
    return (tensorflow.TensorShape([32, 32, 3]), tensorflow.TensorShape([]))

  @property
  def output_types(self):
    return dtypes.uint8, dtypes.uint8
