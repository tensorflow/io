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
"""_IODataset and _IOStreamDataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

class _IOStreamDataset(tf.compat.v2.data.Dataset):
  """_IOStreamDataset"""

  def __init__(self, function, **kwargs):
      capacity = kwargs.get("capacity", 4096)
      dataset = tf.compat.v2.data.Dataset.range(0, sys.maxsize, capacity)
      dataset = dataset.map(lambda index: function(index, index+capacity))
      dataset = dataset.apply(
          tf.data.experimental.take_while(
              lambda v: tf.greater(tf.shape(v)[0], 0)))
      dataset = dataset.unbatch()

      self._function = function
      self._dataset = dataset
      super(_IOStreamDataset, self).__init__(
          self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec

class _IODataset(_IOStreamDataset):
  """_IODataset"""

  def __init__(self, function, **kwargs):
    super(_IODataset, self).__init__(function, **kwargs)

class _IODatasetComponentFunction(object):
  def __init__(self, function, resource, component, shape, dtype):
    self._function = function
    self._resource = resource
    self._component = component
    self._shape = tf.TensorShape([None]).concatenate(shape[1:])
    self._dtype = dtype
  def __call__(self, start, stop):
    return self._function(
        self._resource, start=start, stop=stop,
        component=self._component, shape=self._shape, dtype=self._dtype)
