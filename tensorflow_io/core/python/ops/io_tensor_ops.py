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
"""_IOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class _IOTensorMeta(property):
  """_IOTensorMeta is a decorator that is viewable to __repr__"""
  pass

class _IOTensorDataset(tf.compat.v2.data.Dataset):
  """_IOTensorDataset"""

  def __init__(self, spec, resource, function):
    components = tf.nest.flatten(spec)

    start = 0
    stop = components[0].shape[0]
    capacity = 4096
    entry_start = list(range(start, stop, capacity))
    entry_stop = entry_start[1:] + [stop]

    dataset = tf.compat.v2.data.Dataset.from_tensor_slices((
        tf.constant(entry_start, tf.int64),
        tf.constant(entry_stop, tf.int64)))

    components = [(component, e) for component, e in enumerate(components)]
    components = [
        dataset.map(
            lambda start, stop: function(
                resource,
                start, stop, 1,
                component=component,
                shape=e.shape,
                dtype=e.dtype)) for (component, e) in components]
    dataset = tf.compat.v2.data.Dataset.zip(
        tf.nest.pack_sequence_as(spec, components))
    dataset = dataset.unbatch()

    self._dataset = dataset
    self._resource = resource
    self._function = function
    super(_IOTensorDataset, self).__init__(
        self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec

class _IOTensor(object):
  """_IOTensor"""

  def __init__(self,
               spec,
               internal=False):
    if not internal:
      raise ValueError("IOTensor constructor is private; please use one "
                       "of the factory methods instead (e.g., "
                       "IOTensor.from_tensor())")
    self._spec = spec
    super(_IOTensor, self).__init__()

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
            ) if isinstance(v, _IOTensorMeta)])
    return "<%s: spec=%s%s>" % (
        self.__class__.__name__, self.spec, meta)

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
    return _IOTensorDataset(
        self.spec, self._resource, self._function)

class BaseIOTensor(_IOTensor):
  """BaseIOTensor

  A `BaseIOTensor` is a basic `IOTensor` with only one component.
  It is associated with a `Tensor` of `shape` and `dtype`, with
  data backed by IO. It is the building block for `IOTensor`.
  For example, a `CSVIOTensor` consists of multiple `BaseIOTensor`
  where each one is a column of the CSV.

  All `IOTensor` types are either a subclass of `BaseIOTensor`,
  or are a composite of a collection of `BaseIOTensor`.

  The additional properties exposed by `BaseIOTensor` are `shape`
  and `dtype` associated with counterparts in `Tensor`.
  """

  def __init__(self,
               spec,
               resource,
               function,
               component=0,
               internal=False):
    self._resource = resource
    self._function = function
    self._component = component
    super(BaseIOTensor, self).__init__(
        spec, internal=internal)

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def shape(self):
    """Returns the `TensorShape` that represents the shape of the tensor."""
    return self.spec.shape

  @property
  def dtype(self):
    """Returns the `dtype` of elements in the tensor."""
    return self.spec.dtype

  #=============================================================================
  # Indexing & Slicing
  #=============================================================================
  def __getitem__(self, key):
    """Returns the specified piece of this IOTensor."""
    # Find out the indices based on length and key,
    # based on python slice()'s indices method:
    index = key if isinstance(key, slice) else slice(key, key + 1)
    (start, stop, step) = index.indices(self.shape[0])
    if start >= self.shape[0]:
      raise IndexError("index %s is out of range" % key)
    item = self._function(
        self._resource,
        start, stop, step,
        component=self._component,
        shape=self.spec.shape, dtype=self.spec.dtype)
    return tf.squeeze(item, axis=[0]) if (stop == start + 1) else item

  def __len__(self):
    """Returns the total number of items of this IOTensor."""
    return self.shape[0]

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

class _TableIOTensor(_IOTensor):
  """_TableIOTensor"""

  def __init__(self,
               spec,
               columns,
               resource,
               function,
               internal=False):
    self._columns = columns
    self._resource = resource
    self._function = function
    super(_TableIOTensor, self).__init__(
        spec, internal=internal)

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def columns(self):
    """The names of columns"""
    return self._columns

  def __call__(self, column):
    """Return a BaseIOTensor with column named `column`"""
    component = self.columns.index(
        next(e for e in self.columns if e == column))
    spec = tf.nest.flatten(self.spec)[component]
    return BaseIOTensor(
        spec, self._resource, self._function,
        component=component, internal=True)

class _SeriesIOTensor(_IOTensor):
  """_SeriesIOTensor"""

  def __init__(self,
               spec,
               resource,
               function,
               internal=False):
    self._resource = resource
    self._function = function
    super(_SeriesIOTensor, self).__init__(
        spec, internal=internal)

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def index(self):
    """The index column of the series"""
    return BaseIOTensor(
        self.spec[0], self._resource, self._function,
        component=0, internal=True)

  @property
  def value(self):
    """The value column of the series"""
    return BaseIOTensor(
        self.spec[1], self._resource, self._function,
        component=1, internal=True)
