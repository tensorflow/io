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
"""tensorflow_io.experimental.IODataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_io.core.python.ops import io_dataset
from tensorflow_io.core.python.experimental import libsvm_dataset_ops
from tensorflow_io.core.python.experimental import image_dataset_ops

class IODataset(io_dataset.IODataset):
  """IODataset"""

  #=============================================================================
  # Factory Methods
  #=============================================================================

  @classmethod
  def from_libsvm(cls,
                  filename,
                  num_features,
                  dtype=None,
                  label_dtype=None,
                  compression_type='',
                  **kwargs):
    """Creates an `IODataset` from a libsvm file.

    Args:
      filename: A `tf.string` tensor containing one or more filenames.
      num_features: The number of features.
      dtype(Optional): The type of the output feature tensor.
        Default to tf.float32.
      label_dtype(Optional): The type of the output label tensor.
        Default to tf.int64.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IODataset`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromLibSVM")):
      return libsvm_dataset_ops.LibSVMIODataset(
          filename, num_features,
          dtype=dtype, label_dtype=label_dtype,
          compression_type=compression_type,
          internal=True, **kwargs)

  @classmethod
  def from_tiff(cls,
                filename,
                **kwargs):
    """Creates an `IODataset` from a TIFF file.

    Args:
      filename: A string, the filename of a TIFF file.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IODataset`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromTIFF")):
      return image_dataset_ops.TIFFIODataset(
          filename, internal=True)
