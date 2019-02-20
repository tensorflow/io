# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""LMDBDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
from tensorflow import dtypes
from tensorflow.compat.v1 import data
from tensorflow_io import _load_library
lmdb_ops = _load_library('_lmdb_ops.so')

class LMDBDataset(data.Dataset):
  """A LMDB Dataset that reads the lmdb file."""

  def __init__(self, filenames):
    """Create a `LMDBDataset`.

    `LMDBDataset` allows a user to read data from a mdb file as
    (key value) pairs sequentially.

    For example:
    ```python
    tf.enable_eager_execution()
    dataset = LMDBDataset("/foo/bar.mdb")
    # Prints the (key, value) pairs inside a lmdb file.
    for key, value in dataset:
      print(key, value)
    ```
    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    self._filenames = tensorflow.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    super(LMDBDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return lmdb_ops.lmdb_dataset(
        self._filenames,
        (dtypes.string, dtypes.string),
        (tensorflow.TensorShape([]), tensorflow.TensorShape([])))

  @property
  def output_classes(self):
    return tensorflow.Tensor, tensorflow.Tensor

  @property
  def output_shapes(self):
    return (tensorflow.TensorShape([]), tensorflow.TensorShape([]))

  @property
  def output_types(self):
    return dtypes.string, dtypes.string
