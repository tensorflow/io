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
"""Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import inspect

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

class _Dataset(tf.compat.v2.data.Dataset):
  """"_Dataset"""

  @abc.abstractmethod
  def _inputs(self):
    """Returns a list of the input datasets of the dataset."""

    raise NotImplementedError("Dataset._inputs")

  @abc.abstractproperty
  def _element_structure(self):
    """The structure of an element of this dataset.
    Returns:
      A `Structure` object representing the structure of an element of this
      dataset.
    """
    raise NotImplementedError("Dataset._element_structure")

  def rebatch(self, batch_size, batch_mode=""):
    def _apply_fn(dataset):
      return _AdjustBatchDataset(dataset, batch_size, batch_mode)

    return self.apply(_apply_fn)

class _AdjustBatchDataset(_Dataset):
  """AdjustBatchDataset"""

  def __init__(self, input_dataset, batch_size, batch_mode=""):
    """Create a AdjustBatchDataset."""
    self._input_dataset = input_dataset
    self._batch_size = batch_size
    self._batch_mode = batch_mode
    self._structure = input_dataset._element_structure._unbatch()._batch(None) # pylint: disable=protected-access
    variant_tensor = core_ops.adjust_batch_dataset(
        self._input_dataset._variant_tensor, # pylint: disable=protected-access
        batch_size=self._batch_size,
        batch_mode=self._batch_mode,
        output_types=self._structure._flat_types, # pylint: disable=protected-access
        output_shapes=self._structure._flat_shapes) # pylint: disable=protected-access

    super(_AdjustBatchDataset, self).__init__(variant_tensor)

  def _inputs(self):
    return [self._input_dataset]

  @property
  def _element_structure(self):
    return self._structure


class _DatasetAdapter(_Dataset):
  """_DatasetAdapter"""

  def __init__(self, dataset):
    self._dataset = dataset
    super(_DatasetAdapter, self).__init__(self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return self._dataset._inputs() # pylint: disable=protected-access

  @property
  def _element_structure(self):
    return self._dataset._element_structure # pylint: disable=protected-access

def dataset_decorator(func):
  @functools.wraps(func)
  def f(*args, **kwargs):
    v = func(*args, **kwargs)
    if ((isinstance(v, tf.compat.v2.data.Dataset)) and
        (not isinstance(v, _DatasetAdapter))):
      return _DatasetAdapter(v)
    return v
  return f

def create_dataste_class():
  """create_dataste_class"""
  for k, f in inspect.getmembers(
      tf.compat.v2.data.Dataset,
      lambda f: inspect.isfunction(f) or inspect.ismethod(f)):
    if k not in _Dataset.__dict__:
      # alternative: inspect.ismethod(f) and f.__self__ is not None
      if ((k in tf.compat.v2.data.Dataset.__dict__) and
          (isinstance(tf.compat.v2.data.Dataset.__dict__[k], classmethod))):
        setattr(_Dataset, k, classmethod(dataset_decorator(f)))
      elif ((k in tf.compat.v2.data.Dataset.__dict__) and
            (isinstance(tf.compat.v2.data.Dataset.__dict__[k], staticmethod))):
        setattr(_Dataset, k, staticmethod(dataset_decorator(f)))
      else:
        setattr(_Dataset, k, dataset_decorator(f))
  return _Dataset
