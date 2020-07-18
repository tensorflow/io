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

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class MNISTLabelIODataset(tf.data.Dataset):
    """A MNISTLabelIODataset"""

    def __init__(self, filename):
        """Create a MNISTLabelDataset.
        
        Args:
            filename: A `tf.string` tensor containing filename.
        """
        _, compression = core_ops.io_file_info(filename)
        dataset = tf.data.FixedLengthRecordDataset(
            filename, 1, header_bytes=8, compression_type=compression
        )
        dataset = dataset.map(lambda e: tf.io.decode_raw(e, tf.uint8))
        dataset = dataset.unbatch()

        self._dataset = dataset
        super().__init__(
            self._dataset._variant_tensor
        )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec


class MNISTImageIODataset(tf.data.Dataset):
    """A MNISTImageIODataset
  """

    def __init__(self, filename):
        """Create a MNISTImageDataset.
        
        Args:
            filename: A `tf.string` tensor containing filename.
        """
        _, compression = core_ops.io_file_info(filename)
        rows = tf.io.decode_raw(
            core_ops.io_file_read(filename, 8, 4, compression=compression),
            tf.int32,
            little_endian=False,
        )
        cols = tf.io.decode_raw(
            core_ops.io_file_read(filename, 12, 4, compression=compression),
            tf.int32,
            little_endian=False,
        )
        lens = rows[0] * cols[0]

        dataset = tf.data.FixedLengthRecordDataset(
            filename,
            tf.cast(lens, tf.int64),
            header_bytes=16,
            compression_type=compression,
        )
        dataset = dataset.map(lambda e: tf.io.decode_raw(e, tf.uint8))
        dataset = dataset.map(lambda e: tf.reshape(e, tf.concat([rows, cols], axis=0)))

        self._dataset = dataset
        super().__init__(
            self._dataset._variant_tensor
        )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec


def MNISTIODataset(images=None, labels=None, internal=True):
    """MNISTIODataset"""
    assert internal, (
        "MNISTIODataset constructor is private; please use one "
        "of the factory methods instead (e.g., "
        "IODataset.from_mnist())"
    )

    assert (
        images is not None or labels is not None
    ), "images and labels could not be all None"

    images_dataset = MNISTImageIODataset(images) if images is not None else None

    labels_dataset = MNISTLabelIODataset(labels) if labels is not None else None

    if images is None:
        return labels_dataset
    if labels is None:
        return images_dataset

    return tf.data.Dataset.zip((images_dataset, labels_dataset))
