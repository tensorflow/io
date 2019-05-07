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
"""Parquet Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
from tensorflow.compat.v1 import data
from tensorflow_io import _load_library
parquet_ops = _load_library('_parquet_ops.so')

class ParquetDataset(data.Dataset):
  """A Parquet Dataset that reads the parquet file."""

  def __init__(self, filename, columns, dtypes=None, batch=None):
    """Create a `ParquetDataset`.

    `ParquetDataset` allows a user to read data from a parquet file.
    For example:

    ```python
    dataset = tf.contrib.parquet.ParquetDataset(
        "/foo/bar.parquet", [0, 1], (tf.bool, tf.int32))
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # Prints the rows of the result set of the column [0, 1].
    while True:
      try:
        print(sess.run(next_element))
      except tf.errors.OutOfRangeError:
        break
    ```

    Args:
      filename: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      columns: A 0-D or 1-D `tf.int32` tensor containing the columns to extract.
      dtypes: A tuple of `tf.DType` objects representing the types of the
        columns returned.
    """
    self._data_input = parquet_ops.parquet_input(
        filename, ["none", "gz"], columns=columns)
    self._columns = columns
    self._dtypes = dtypes
    self._batch = 0 if batch is None else batch
    super(ParquetDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return parquet_ops.parquet_dataset(
        self._data_input,
        self._batch,
        output_types=self.output_types,
        output_shapes=self.output_shapes)

  @property
  def output_classes(self):
    return tuple([tensorflow.Tensor for _ in self._columns])

  @property
  def output_shapes(self):
    return tuple(
        [tensorflow.TensorShape([]) for _ in self._columns]
    ) if self._batch is None else tuple(
        [tensorflow.TensorShape([None]) for _ in self._columns]
    )

  @property
  def output_types(self):
    return self._dtypes
