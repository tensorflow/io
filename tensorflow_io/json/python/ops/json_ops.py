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
"""JSONDataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops
from tensorflow_io.core.python.ops import core_ops

def list_json_columns(filename):
  """list_json_columns"""
  if not tf.executing_eagerly():
    raise NotImplementedError("list_json_columns only support eager mode")
  columns, dtypes = core_ops.io_list_json_columns(filename)
  entries = zip(tf.unstack(columns), tf.unstack(dtypes))
  return dict([(column.numpy().decode(), tf.TensorSpec(
      tf.TensorShape([None]),
      dtype.numpy().decode(),
      column.numpy().decode())) for (
          column, dtype) in entries])

def read_json(filename, column):
  """read_json"""
  return core_ops.io_read_json(
      filename, column.name, dtype=column.dtype)

class JSONDataset(data_ops.BaseDataset):
  """A JSONLabelDataset. JSON (JavaScript Object Notation) is a lightweight data-interchange format.
  """

  def __init__(self, filename, columns, **kwargs):
    """Create a JSONLabelDataset.

    Args:
      filename: A string containing one or more filenames.
      columns: A list of strings containing the columns to extract.
    """
    if not tf.executing_eagerly():
      self._dtypes = kwargs.get("dtype")
    else:
      all_columns = list_json_columns(filename)
      for column in columns:
        if column not in all_columns:
          raise ValueError(
              "There is no column named {} in the {}".format(filename, column))
      self._dtypes = [all_columns[column].dtype for column in columns]

    self._shapes = [tf.TensorShape([None])] * len(columns)

    datasets = []
    for i, column in enumerate(columns):
      datasets.append(data_ops.Dataset.from_tensors(
          core_ops.io_read_json(filename, column, dtype=self._dtypes[i])))

    self._dataset = tf.compat.v2.data.Dataset.zip(tuple(datasets))

    super(JSONDataset, self).__init__(
        self._dataset._variant_tensor, self._dtypes, self._shapes) # pylint: disable=protected-access
