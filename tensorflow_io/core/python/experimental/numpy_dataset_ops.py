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
"""NumpyIODataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

class NumpyIODataset(tf.data.Dataset):
  """NumpyIODataset"""

  def __init__(self, a, internal=True):
    """NumpyIODataset."""
    with tf.name_scope("NumpyIODataset"):
      assert internal

      flatten = tf.nest.flatten(a)
      assert all([array.shape[0]==flatten[0].shape[0] for array in flatten])

      def p(array):
        address, _ = array.__array_interface__['data']
        shape = array.shape
        dtype = tf.as_dtype(array.dtype)
        return address, shape, dtype
      params = [p(array) for array in flatten]
      stop = tf.constant(flatten[0].shape[0], tf.int64)
      step = 1024
      indices_start = tf.data.Dataset.range(0, stop, step)
      indices_stop = indices_start.skip(1).concatenate(
          tf.data.Dataset.from_tensor_slices([stop]))
      dataset = tf.data.Dataset.zip((indices_start, indices_stop))
      def f(start, stop):
        return tf.nest.pack_sequence_as(a, [g(start, stop, address, shape, dtype) for address, shape, dtype in params])
      def g(start, stop, address, shape, dtype):
        return core_ops.io_numpy_read(
            address=address, filename="", array="", shape=shape,
            start=start, stop=stop, dtype=dtype)
      dataset = dataset.map(f)
      dataset = dataset.unbatch()

      self._dataset = dataset
      self._holder = [np.array(array, copy=False) for array in flatten]
      super(NumpyIODataset, self).__init__(
          self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec
