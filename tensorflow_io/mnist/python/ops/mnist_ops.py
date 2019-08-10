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
from tensorflow_io.core.python.ops import data_ops
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.ops import archive_ops

class MNISTLabelDataset(data_ops.BaseDataset):
  """A MNISTLabelDataset
  """

  def __init__(self, filename, batch=None):
    """Create a MNISTLabelDataset.
    Args:
      filename: A `tf.string` tensor containing filename.
    """
    f, entries = archive_ops.list_archive_entries(
        filename, ["none", "gz"])
    memory = archive_ops.read_archive(filename, f, entries)
    labels, = core_ops.read_mnist_label(
        filename, memory=memory, metadata="",
        start=0, stop=-1, dtypes=[tf.uint8])

    dataset = data_ops.BaseDataset.from_tensors(labels)

    batch = 0 if batch is None else batch
    dtypes = [tf.uint8]
    shapes = [
        tf.TensorShape([])] if batch == 0 else [
            tf.TensorShape([None])]
    if batch == 0:
      dataset = dataset.apply(tf.data.experimental.unbatch())
    else:
      dataset = dataset.apply(data_ops.rebatch(batch))

    super(MNISTLabelDataset, self).__init__(
        dataset._variant_tensor, dtypes, shapes) # pylint: disable=protected-access

class MNISTImageDataset(data_ops.BaseDataset):
  """A MNISTImageDataset
  """

  def __init__(self, filename, batch=None):
    """Create a MNISTImageDataset.
    Args:
      filename: A `tf.string` tensor containing filename.
    """
    f, entries = archive_ops.list_archive_entries(
        filename, ["none", "gz"])
    memory = archive_ops.read_archive(filename, f, entries)
    images, = core_ops.read_mnist_image(
        filename, memory=memory, metadata="",
        start=0, stop=-1, dtypes=[tf.uint8])

    dataset = data_ops.BaseDataset.from_tensors(images)

    batch = 0 if batch is None else batch
    dtypes = [tf.uint8]
    shapes = [
        tf.TensorShape([None, None])] if batch == 0 else [
            tf.TensorShape([None, None, None])]
    if batch == 0:
      dataset = dataset.apply(tf.data.experimental.unbatch())
    else:
      dataset = dataset.apply(data_ops.rebatch(batch))

    super(MNISTImageDataset, self).__init__(
        dataset._variant_tensor, dtypes, shapes) # pylint: disable=protected-access

def MNISTDataset(image_filename, label_filename, batch=None):
  batch = 0 if batch is None else batch
  dataset = data_ops.Dataset.zip((
      MNISTImageDataset(image_filename, 10000),
      MNISTLabelDataset(label_filename, 10000)))
  if batch != 0:
    dataset = dataset.apply(data_ops.rebatch(batch))
  else:
    dataset = dataset.apply(tf.data.experimental.unbatch())
  return dataset
