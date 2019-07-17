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
"""DataFrame"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow_io.text.python.ops import text_ops
from tensorflow_io.core.python.ops import data_ops

class ILocIndexer(object):
  """_iLocIndexer"""
  def __init__(self, df):
    self._df = df
    return

  def __getitem__(self, item):
    data = dict(
        (column, tf.Variable(self._df._data[column][item]) # pylint: disable=protected-access
        ) for column in self._df.columns)
    return DataFrame(data, columns=list(self._df.columns))


class DataFrame(object):
  """DataFrame

  NOTE: Experimental and eager only!
  """

  def __init__(self, data=None, columns=None):
    """Create a DataFrame."""
    if not tf.executing_eagerly():
      raise NotImplementedError("DataFrame only supports eager mode")
    self._data = data
    self._columns = columns
    self._index = np.arange(data[columns[0]].shape[0].value)
    self._iloc = _ILocIndexer(self)

  @property
  def columns(self):
    return self._columns

  @property
  def index(self):
    return self._index

  @property
  def shape(self):
    return len(self.index), len(self.columns)

  @property
  def iloc(self):
    return self._iloc

  def __len__(self):
    return len(self._index)

  def __iter__(self):
    return iter(self._index)

  def __getitem__(self, item):
    return self._data[item]

  def __getattr__(self, name):
    if name in self._columns:
      return self.get(name)
    return object.__getattribute__(self, name)

  def __eq__(self, other):
    data = dict(
        (column, tf.Variable(tf.math.equal(self._data[column], other))
        ) for column in self.columns)
    return DataFrame(data, columns=list(self.columns))

  def keys(self):
    return self._columns

  def get(self, key, default=None):
    try:
      return DataFrame({key: tf.Variable(self._data[key])}, columns=[key])
    except (KeyError, ValueError, IndexError):
      return default

  def pop(self, item):
    e = self._data.pop(item)
    self._columns.remove(item)
    return DataFrame({item: e}, columns=[item])

  def head(self, n=5):
    return self.iloc[:n]

  def tail(self, n=5):
    if n == 0:
      return self.iloc[0:0]
    return self.iloc[-n:]

  def split(self, func):
    indices_x, indices_y = func(self._index)
    data_x, data_y = {}, {}
    for column in self._columns:
      data_x[column] = tf.Variable(tf.gather(self._data[column], indices_x))
      data_y[column] = tf.Variable(tf.gather(self._data[column], indices_y))
    return DataFrame(
        data_x, list(self._columns)), DataFrame(
            data_y, list(self._columns))

  @staticmethod
  def from_csv(filename, header=0):
    """from_csv"""
    dataset, specs = text_ops.from_csv(filename, header=header)
    values = data_ops._dataset_to_tensors(dataset.batch(32))  # pylint: disable=protected-access
    data = {}
    for (value, spec) in zip(values, specs):
      # Assert shape are compatible
      # shape = tf.TensorShape([None]).concatenate(spec.shape)
      # assert shape.is_compatible_with(value.shape)
      # assert spec.dtype == value.dtype
      # assert value.shape == values[0].shape
      data[spec.name] = tf.Variable(value)
    columns = [spec.name for spec in specs]
    return DataFrame(data, columns=columns)
