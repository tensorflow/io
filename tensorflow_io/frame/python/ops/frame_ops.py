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

  def __len__(self):
    return self.shape[0]

  def __iter__(self):
    for key in self.columns:
      yield self._data[key]

  def __getitem__(self, item):
    if item in self.columns:
      return self._data[item]
    raise NotImplementedError

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
