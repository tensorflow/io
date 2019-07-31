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

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

class _AdjustBatchDataset(tf.compat.v2.data.Dataset):
  """AdjustBatchDataset"""

  def __init__(self, input_dataset, batch_size, batch_mode=""):
    """Create a AdjustBatchDataset."""
    self._input_dataset = input_dataset
    self._batch_size = batch_size
    self._batch_mode = batch_mode

    self._structure = input_dataset._element_structure._unbatch()._batch(None) # pylint: disable=protected-access

    variant_tensor = core_ops.adjust_batch_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
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

def rebatch(batch_size, batch_mode=""):
  def _apply_fn(dataset):
    return _AdjustBatchDataset(dataset, batch_size, batch_mode)

  return _apply_fn

# Note: BaseDataset could be used by Dataset implementations
# that does not utilize DataInput implementation.
class BaseDataset(tf.compat.v2.data.Dataset):
  """A Base Dataset"""

  def __init__(self, variant, dtypes, shapes):
    """Create a Base Dataset."""
    self._dtypes = dtypes
    self._shapes = shapes
    super(BaseDataset, self).__init__(variant)

  def _inputs(self):
    return []

  @property
  def _element_structure(self):
    e = [
        tf.data.experimental.TensorStructure(
            p, q.as_list()) for (p, q) in zip(
                self._dtypes, self._shapes)
    ]
    if len(e) == 1:
      return e[0]
    return tf.data.experimental.NestedStructure(tuple(e))

class Dataset(BaseDataset):
  """A Dataset that takes DataInput"""

  def __init__(self, fn, data_input, batch, dtypes, shapes):
    """Create a Dataset."""
    self._fn = fn
    self._data_input = data_input
    self._batch = 0 if batch is None else batch
    self._dtypes = dtypes
    self._shapes = shapes
    super(Dataset, self).__init__(fn(
        self._data_input,
        self._batch,
        output_types=self._dtypes,
        output_shapes=self._shapes), self._dtypes, self._shapes)
