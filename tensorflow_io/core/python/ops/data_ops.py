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

import tensorflow as tf

# Note: BaseDataset could be used by Dataset implementations
# that does not utilize DataInput implementation.
class BaseDataset(tf.compat.v2.data.Dataset):
  """A Base Dataset"""

  def __init__(self, variant, dtypes, shapes):
    """Create a Base Dataset."""
    self._dtypes = dtypes
    self._shapes = shapes
    super().__init__(variant)

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    e = [
        tf.TensorSpec(
            p.as_list(), q) for (p, q) in zip(
                self._shapes, self._dtypes)
    ]
    if len(e) == 1:
      return e[0]
    return tuple(e)

class Dataset(BaseDataset):
  """A Dataset that takes DataInput"""

  def __init__(self, fn, data_input, batch, dtypes, shapes):
    """Create a Dataset."""
    self._fn = fn
    self._data_input = data_input
    self._batch = 0 if batch is None else batch
    self._dtypes = dtypes
    self._shapes = shapes
    super().__init__(fn(
        self._data_input,
        self._batch,
        output_types=self._dtypes,
        output_shapes=self._shapes), self._dtypes, self._shapes)
