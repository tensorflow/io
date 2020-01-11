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

      if not isinstance(a, str):
        entries = a

        def p(entry):
          address, _ = entry.__array_interface__['data']
          shape = entry.shape
          dtype = tf.as_dtype(entry.dtype)
          return address, "", "", shape, dtype
      else:
        filename = a
        arrays, shapes, dtypes = core_ops.io_numpy_info(filename=filename)
        arrays = tf.unstack(arrays)
        shapes = tf.unstack(shapes)
        dtypes = tf.unstack(dtypes)
        dtypes = [tf.as_dtype(dtype.numpy()) for dtype in dtypes]

        entries = list(zip(shapes, dtypes, arrays))
        entries = [tf.TensorSpec(shape, dtype, array) for (
            shape, dtype, array) in entries]

        indices = None
        if all([e.numpy().decode().startswith("arr_") for e in arrays]):
          try:
            indices = [int(e.numpy()[4:]) for e in arrays]
          except ValueError:
            pass
        if indices is not None:
          values = list(indices)
          values.sort()
          if not all([k == v for k, v in enumerate(values)]):
            indices = None

        # if indices is continuously, then construct a tuple, otherwise a dict.
        if indices is not None:
          entries = dict(zip(indices, entries))
          entries = tuple([entries[index] for index in sorted(indices)])
        else:
          indices = [index.numpy().decode() for index in tf.unstack(arrays)]
          entries = dict(zip(indices, entries))

        def p(entry):
          return 0, filename, entry.name, entry.shape, entry.dtype

      flatten = tf.nest.flatten(entries)
      assert all([entry.shape[0] == flatten[0].shape[0] for entry in flatten])

      params = [p(entry) for entry in flatten]

      def f(start, stop):
        return tf.nest.pack_sequence_as(
            entries,
            [
                core_ops.io_numpy_read(
                    address=address, filename=filename, array=array,
                    shape=shape, start=start, stop=stop, dtype=dtype
                ) for address, filename, array, shape, dtype in params])

      step = 1024
      total = tf.constant(flatten[0].shape[0], tf.int64)
      indices_start = tf.data.Dataset.range(0, total, step)
      indices_stop = indices_start.skip(1).concatenate(
          tf.data.Dataset.from_tensor_slices([total]))
      dataset = tf.data.Dataset.zip((indices_start, indices_stop))
      dataset = dataset.map(f)
      dataset = dataset.unbatch()

      self._dataset = dataset
      if isinstance(a, str):
        self._holder = [np.array(entry, copy=False) for entry in flatten]
      super(NumpyIODataset, self).__init__(
          self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec
