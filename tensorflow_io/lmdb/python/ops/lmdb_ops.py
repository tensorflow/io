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

import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops as data_ops
from tensorflow_io.core.python.ops import core_ops as lmdb_ops

class LMDBDataset(data_ops.Dataset):
  """A LMDB Dataset that reads the lmdb file."""

  def __init__(self, filename, batch=None):
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
      filename: A `tf.string` tensor containing one or more filenames.
    """
    batch = 0 if batch is None else batch
    dtypes = [tf.string, tf.string]
    shapes = [
        tf.TensorShape([]), tf.TensorShape([])] if batch == 0 else [
            tf.TensorShape([None]), tf.TensorShape([None])]
    super(LMDBDataset, self).__init__(
        lmdb_ops.lmdb_dataset,
        lmdb_ops.lmdb_input(filename),
        batch, dtypes, shapes)
