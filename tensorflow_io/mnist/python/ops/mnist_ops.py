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

import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops as data_ops
from tensorflow_io.core.python.ops import core_ops as mnist_ops

class MNISTLabelDataset(data_ops.Dataset):
  """A MNISTLabelDataset
  """

  def __init__(self, filename, batch=None):
    """Create a MNISTLabelDataset.
    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    batch = 0 if batch is None else batch
    dtypes = [tf.uint8]
    shapes = [
        tf.TensorShape([])] if batch == 0 else [
            tf.TensorShape([None])]
    super(MNISTLabelDataset, self).__init__(
        mnist_ops.mnist_label_dataset,
        mnist_ops.mnist_label_input(filename, ["none", "gz"]),
        batch, dtypes, shapes)

class MNISTImageDataset(data_ops.Dataset):
  """A MNISTImageDataset
  """

  def __init__(self, filename, batch=None):
    """Create a MNISTImageDataset.
    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    batch = 0 if batch is None else batch
    dtypes = [tf.uint8]
    shapes = [
        tf.TensorShape([None, None])] if batch == 0 else [
            tf.TensorShape([None, None, None])]
    super(MNISTImageDataset, self).__init__(
        mnist_ops.mnist_image_dataset,
        mnist_ops.mnist_image_input(filename, ["none", "gz"]),
        batch, dtypes, shapes)

def MNISTDataset(image_filename, label_filename, batch=None):
  return data_ops.Dataset.zip((
      MNISTImageDataset(image_filename, batch),
      MNISTLabelDataset(label_filename, batch)))
