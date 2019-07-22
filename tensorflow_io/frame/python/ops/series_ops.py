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
"""Series"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

class Series(object):
  """Series"""

  def __init__(self, data=None, name=None):
    if not tf.executing_eagerly():
      raise NotImplementedError("Series only supports eager mode")
    self._name = name
    data = tf.convert_to_tensor(data)
    if tf.rank(data) > 1:
      data = tf.squeeze(data, axis=np.arange(1, tf.rank(data)))
    self._data = tf.Variable(data)
    self._index = np.arange(self._data.shape.as_list()[0])

  @property
  def name(self):
    return self._name

  @property
  def index(self):
    return self._index

  @property
  def dtype(self):
    return self._data.dtype

  @property
  def shape(self):
    return (len(self.index),)

  @property
  def ndim(self):
    return 1

  @property
  def values(self):
    return self._data.numpy()

  def __len__(self):
    return len(self.index)

  def __iter__(self):
    for key in self.index:
      yield self._data[key].numpy()

  def __getitem__(self, key):
    # self._data[key].numpy()
    raise NotImplementedError

  def __setitem__(self, key, value):
    # self._data[key].assign(value)
    raise NotImplementedError

  def __eq__(self, other):
    data = tf.math.equal(self._data, other)
    return Series(data, name=self.name)
