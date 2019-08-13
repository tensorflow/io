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

import sys

import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops
from tensorflow_io.core.python.ops import core_ops

def read_lmdb(filename):
  """read_lmdb"""
  resource, _, _ = core_ops.init_lmdb(filename, memory="", metadata="")
  value, key = core_ops.next_lmdb(resource, -1)
  return key, value

class LMDBDataset(data_ops.BaseDataset):
  """A LMDB Dataset that reads the lmdb file."""

  def __init__(self, filename, **kwargs):
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
      filename: A `tf.string` tensor containing filename.
    """
    dtypes = [tf.string, tf.string]
    shapes = [tf.TensorShape([None]), tf.TensorShape([None])]
    capacity = kwargs.get("capacity", 65536)
    resource, _, _ = core_ops.init_lmdb(filename, memory="", metadata="")
    dataset = data_ops.BaseDataset.range(
        0, sys.maxsize, capacity).map(
            lambda i: core_ops.next_lmdb(resource, capacity)).apply(
                tf.data.experimental.take_while(
                    lambda (v, k): tf.shape(v)[0] > 0)).map(
                        lambda (v, k): (k, v))
    self._resource = resource
    self._dataset = dataset

    super(LMDBDataset, self).__init__(
        self._dataset._variant_tensor, dtypes, shapes) # pylint: disable=protected-access
