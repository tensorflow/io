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
"""MNISTIODataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.ops import data_ops

class MNISTLabelIODataset(data_ops.Dataset):
  """A MNISTLabelIODataset"""

  def __init__(self, filename):
    """Create a MNISTLabelDataset.
    Args:
      filename: A `tf.string` tensor containing filename.
    """
    dtypes = [tf.uint8]
    shapes = [tf.TensorShape([])]
    super(MNISTLabelIODataset, self).__init__(
        core_ops.io_mnist_label_dataset,
        core_ops.io_mnist_label_input(filename, ["none", "gz"]),
        0, dtypes, shapes)

class MNISTImageIODataset(data_ops.Dataset):
  """A MNISTImageIODataset
  """

  def __init__(self, filename):
    """Create a MNISTImageDataset.
    Args:
      filename: A `tf.string` tensor containing filename.
    """
    dtypes = [tf.uint8]
    shapes = [tf.TensorShape([None, None])]
    super(MNISTImageIODataset, self).__init__(
        core_ops.io_mnist_image_dataset,
        core_ops.io_mnist_image_input(filename, ["none", "gz"]),
        0, dtypes, shapes)

def MNISTIODataset(images=None, labels=None, internal=True):
  """MNISTIODataset"""
  assert internal, ("MNISTIODataset constructor is private; please use one "
                    "of the factory methods instead (e.g., "
                    "IODataset.from_avro())")

  assert images is not None or labels is not None, (
      "images and labels could not be all None")

  images_dataset = MNISTImageIODataset(
      images) if images is not None else None

  labels_dataset = MNISTLabelIODataset(
      labels) if labels is not None else None

  if images is None:
    return labels_dataset
  if labels is None:
    return images_dataset

  return data_ops.Dataset.zip((images_dataset, labels_dataset))
