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
"""_BaseIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class _BaseIOTensorMeta(property):
  """_BaseIOTensorMeta is a decorator that is viewable to __repr__"""
  pass

class _BaseIOTensorDataset(tf.compat.v2.data.Dataset):
  """_IOTensorDataset"""

  def __init__(self, spec, resource, function):
    start = 0
    stop = tf.nest.flatten(
        tf.nest.map_structure(lambda e: e.shape, spec))[0][0]
    capacity = 4096
    entry_start = list(range(start, stop, capacity))
    entry_stop = entry_start[1:] + [stop]

    dtype = tf.nest.flatten(
        tf.nest.map_structure(lambda e: e.dtype, spec))
    shape = tf.nest.flatten(
        tf.nest.map_structure(
            lambda e: tf.TensorShape(
                [None]).concatenate(e.shape[1:]), spec))

    dataset = tf.compat.v2.data.Dataset.from_tensor_slices((
        tf.constant(entry_start, tf.int64),
        tf.constant(entry_stop, tf.int64)))
    dataset = dataset.map(
        lambda start, stop: function(
            resource, start, stop, 1, dtype=dtype, shape=shape))
    # Note: tf.data.Dataset consider tuple `(e, )` as one element
    # instead of a sequence. So next `unbatch()` will not work.
    # The tf.stack() below is necessary.
    if len(dtype) == 1:
      dataset = dataset.map(tf.stack)
    dataset = dataset.apply(tf.data.experimental.unbatch())
    self._dataset = dataset
    self._resource = resource
    self._function = function
    super(_BaseIOTensorDataset, self).__init__(
        self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec

class _BaseIOTensor(object):
  """_BaseIOTensor"""

  def __init__(self,
               spec,
               resource,
               function,
               internal=False):
    if not internal:
      raise ValueError("IOTensor constructor is private; please use one "
                       "of the factory methods instead (e.g., "
                       "IOTensor.from_tensor())")
    self._spec = spec
    self._resource = resource
    self._function = function
    super(_BaseIOTensor, self).__init__()

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def spec(self):
    """The `TensorSpec` of values in this tensor."""
    return self._spec

  #=============================================================================
  # String Encoding
  #=============================================================================
  def __repr__(self):
    meta = "".join([", %s=%s" % (
        k, repr(v.__get__(self))) for k, v in self.__class__.__dict__.items(
            ) if isinstance(v, _BaseIOTensorMeta)])
    return "<%s: spec=%s%s>" % (
        self.__class__.__name__, self.spec, meta)


  #=============================================================================
  # Indexing & Slicing
  #=============================================================================
  def __getitem__(self, key):
    """Returns the specified piece of this IOTensor."""
    if isinstance(key, slice):
      start = key.start
      stop = key.stop
      step = key.step
      if start is None:
        start = 0
      if stop is None:
        stop = -1
      if step is None:
        step = 1
    else:
      start = key
      stop = key + 1
      step = 1
    dtype = tf.nest.flatten(
        tf.nest.map_structure(lambda e: e.dtype, self.spec))
    shape = tf.nest.flatten(
        tf.nest.map_structure(lambda e: e.shape, self.spec))
    return tf.nest.pack_sequence_as(self.spec, self._function(
        self._resource,
        start, stop, step,
        dtype=dtype,
        shape=shape))

  def __len__(self):
    """Returns the total number of items of this IOTensor."""
    return tf.nest.flatten(
        tf.nest.map_structure(lambda e: e.shape, self.spec))[0][0]

  #=============================================================================
  # Tensor Type Conversions
  #=============================================================================

  @classmethod
  def from_tensor(cls,
                  tensor,
                  **kwargs):
    """Converts a `tf.Tensor` into a `IOTensor`.

    Examples:

    ```python
    ```

    Args:
      tensor: The `Tensor` to convert.

    Returns:
      A `IOTensor`.

    Raises:
      ValueError: If tensor is not a `Tensor`.
    """
    with tf.name_scope(kwargs.get("name", "IOFromTensor")):
      _ = tensor
      raise NotImplementedError()

  def to_tensor(self, **kwargs):
    """Converts this `IOTensor` into a `tf.Tensor`.

    Example:

    ```python
    ```

    Args:
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `Tensor` with value obtained from this `IOTensor`.
    """
    with tf.name_scope(kwargs.get("name", "IOToTensor")):
      return self.__getitem__(slice(None, None))

  #=============================================================================
  # Dataset Conversions
  #=============================================================================

  def to_dataset(self):
    """Converts this `IOTensor` into a `tf.data.Dataset`.

    Example:

    ```python
    ```

    Args:

    Returns:
      A `tf.data.Dataset` with value obtained from this `IOTensor`.
    """
    return _BaseIOTensorDataset(
        self.spec, self._resource, self._function)

class _ColumnIOTensor(_BaseIOTensor):
  """_ColumnIOTensor"""

  def __init__(self,
               shapes,
               dtypes,
               resource,
               function,
               internal=False):
    shapes = [
        tf.TensorShape(
            [None if dim < 0 else dim for dim in e.numpy() if dim != 0]
        ) for e in tf.unstack(shapes)]
    dtypes = [tf.as_dtype(e.numpy()) for e in tf.unstack(dtypes)]
    spec = [tf.TensorSpec(shape, dtype) for (
        shape, dtype) in zip(shapes, dtypes)]
    assert len(spec) == 1
    spec = spec[0]

    self._shape = spec.shape
    self._dtype = spec.dtype
    super(_ColumnIOTensor, self).__init__(
        spec, resource, function, internal=internal)

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def shape(self):
    """Returns the `TensorShape` that represents the shape of the tensor."""
    return self._shape

  @property
  def dtype(self):
    """Returns the `dtype` of elements in the tensor."""
    return self._dtype

class _TableIOTensor(_BaseIOTensor):
  """_TableIOTensor"""

  def __init__(self,
               shapes,
               dtypes,
               columns,
               filename,
               resource,
               function,
               internal=False):
    shapes = [
        tf.TensorShape(
            [None if dim < 0 else dim for dim in e.numpy() if dim != 0]
        ) for e in tf.unstack(shapes)]
    dtypes = [tf.as_dtype(e.numpy()) for e in tf.unstack(dtypes)]
    columns = [e.numpy().decode() for e in tf.unstack(columns)]
    spec = [tf.TensorSpec(shape, dtype, column) for (
        shape, dtype, column) in zip(shapes, dtypes, columns)]
    if len(spec) == 1:
      spec = spec[0]
    else:
      spec = tuple(spec)
    self._filename = filename
    super(_TableIOTensor, self).__init__(
        spec, resource, function, internal=internal)

  #=============================================================================
  # Accessors
  #=============================================================================

  def columns(self):
    """The `TensorSpec` of column named `name`"""
    return [e.name for e in tf.nest.flatten(self.spec)]

  def shape(self, column):
    """Returns the `TensorShape` shape of `column` in the tensor."""
    return next(e.shape for e in tf.nest.flatten(self.spec) if e.name == column)

  def dtype(self, column):
    """Returns the `dtype` of `column` in the tensor."""
    return next(e.dtype for e in tf.nest.flatten(self.spec) if e.name == column)

  def __call__(self, column):
    """Return a new IOTensor with column named `column`"""
    return self.__class__(self._filename, columns=[column], internal=True) # pylint: disable=no-value-for-parameter
