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

import collections
import numpy as np

import tensorflow as tf
from tensorflow_io.frame.python.ops import series_ops
from tensorflow_io.text.python.ops import text_ops

class _iLocIndexer(object): # pylint: disable=invalid-name
  """_iLocIndexer"""
  def __init__(self, df):
    self._df = df

  def __getitem__(self, item):
    indices = self._df.index[item]
    data = [
        tf.gather(
            self._df._data[column]._data, indices # pylint: disable=protected-access
        ) for column in self._df.columns] # pylint: disable=protected-access
    return DataFrame(data, list(self._df.columns))

class DataFrame(object):
  """DataFrame

  NOTE: Experimental and eager only!
  """
  def __init__(self, data=None, columns=None):
    """Create a DataFrame."""
    if not tf.executing_eagerly():
      raise NotImplementedError("Series only supports eager mode")
    self._data = collections.OrderedDict(
        [(k, series_ops.Series(v, name=k)) for (k, v) in zip(columns, data)])
    self._index = np.arange(data[0].shape.as_list()[0])
    self._iloc = _iLocIndexer(self)

  @property
  def columns(self):
    return self._data.keys()

  @property
  def index(self):
    return self._index

  @property
  def dtypes(self):
    raise NotImplementedError

  @property
  def shape(self):
    return tuple([len(self.index), len(self.columns)])

  @property
  def ndim(self):
    return 2

  @property
  def iloc(self):
    return self._iloc

  @property
  def values(self):
    """values"""
    if len(self.columns) == 1:
      return self._data[self.columns[0]].values
    l = [tf.bool, tf.int32, tf.int64, tf.float32, tf.float64, tf.string]
    dtype = l[
        max([l.index(self._data[column].dtype) for column in self.columns])]
    if dtype != tf.string:
      data = tf.stack([
          tf.cast(self._data[column]._data, dtype) for column in self.columns # pylint: disable=protected-access
      ], axis=1)
      return data.numpy()
    raise NotImplementedError

  def __len__(self):
    return self.shape[0]

  def __iter__(self):
    for key in self.columns:
      yield self._data[key]

  def __getitem__(self, item):
    if isinstance(item, series_ops.Series) and item.dtype == tf.bool:
      data = [
          tf.boolean_mask(
              self._data[column]._data, item._data) for column in self.columns] # pylint: disable=protected-access
      return DataFrame(data, list(self.columns))
    if item in self.columns:
      return self._data[item]
    raise NotImplementedError

  def __setitem__(self, key, value):
    if key in self.columns:
      self._data[key] = series_ops.Series(value, name=key)
      return
    raise NotImplementedError

  def __getattr__(self, name):
    if name in self.columns:
      return self._data[name]
    raise AttributeError("'DataFrame' object has no attribute '%s'" % name)

  def keys(self):
    return self.columns

  def pop(self, item):
    return self._data.pop(item)

  def head(self, n=5):
    return self.iloc[:n]

  def tail(self, n=5):
    if n == 0:
      return self.iloc[0:0]
    return self.iloc[-n:]

  def split(self, func):
    indices_x, indices_y = func(self.index)
    data_x = [
        tf.gather(
            self._data[column]._data, indices_x) for column in self.columns] # pylint: disable=protected-access
    data_y = [
        tf.gather(
            self._data[column]._data, indices_y) for column in self.columns] # pylint: disable=protected-access
    return DataFrame(
        data_x, list(self.columns)), DataFrame(
            data_y, list(self.columns))

  @staticmethod
  def from_csv(filename, header=0):
    """from_csv"""
    dataset, specs = text_ops.from_csv(filename, header=header)
    chunks = [chunk for chunk in dataset.batch(32)]
    data = [tf.concat(item, axis=0) for item in list(zip(*chunks))]
    for (value, spec) in zip(data, specs):
      # Assert shape are compatible
      shape = tf.TensorShape([None]).concatenate(spec.shape)
      assert shape.is_compatible_with(value.shape)
      assert spec.dtype == value.dtype
      assert value.shape == data[0].shape
    columns = [spec.name for spec in specs]
    return DataFrame(data, columns)
