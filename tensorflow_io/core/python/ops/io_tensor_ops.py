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

import sys

import tensorflow as tf

class _IOTensorMeta(property):
  """_IOTensorMeta is a decorator that is viewable to __repr__"""
  pass

class _IOTensorPartitionedFunction(object):
  """PartitionedFunction will translate call to cached Function call"""
  def __init__(self, func, partitions):
    self._func = func
    self._partitions = partitions
    partitions_indices = tf.cumsum(partitions).numpy().tolist()
    self._partitions_start = list([0] + partitions_indices[:-1])
    self._partitions_stop = partitions_indices
    self._tensors = [None for _ in partitions]

  def __call__(self, resource, start, stop):
    indices_start = tf.math.maximum(self._partitions_start, start)
    indices_stop = tf.math.minimum(self._partitions_stop, stop)
    indices_hit = tf.math.less(indices_start, indices_stop)
    indices = tf.squeeze(tf.compat.v2.where(indices_hit), [1])
    items = []
    # TODO: change to tf.while_loop
    for index in indices:
      if self._tensors[index] is None:
        self._tensors[index] = self._func(
            resource,
            self._partitions_start[index],
            self._partitions_stop[index])
      slice_start = indices_start[index] - self._partitions_start[index]
      slice_size = indices_stop[index] - indices_start[index]
      item = tf.slice(self._tensors[index], [slice_start], [slice_size])
      items.append(item)
    return tf.concat(items, axis=0)

class _IOTensorDataset(tf.compat.v2.data.Dataset):
  """_IOTensorDataset"""

  def __init__(self, spec, resource, function, partitions):
    if partitions is None:
      start = 0
      stop = tf.nest.flatten(spec)[0].shape[0]
      capacity = 4096
      entry_start = list(range(start, stop, capacity))
      entry_stop = entry_start[1:] + [stop]
    else:
      partitions = tf.cast(partitions, tf.int64)
      entry_stop = tf.cumsum(partitions)
      entry_start = tf.concat([[0], entry_stop[:-1]], axis=0)
    dataset = tf.compat.v2.data.Dataset.from_tensor_slices((
        tf.constant(entry_start, tf.int64),
        tf.constant(entry_stop, tf.int64)))
    dataset = dataset.map(lambda start, stop: function(resource, start=start, stop=stop))
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
               partitions,
               internal=False):
    # function used for dataset should not be partitioned.
    self._dataset_function = function
    if partitions is not None:
      function = _IOTensorPartitionedFunction(function, partitions)
    self._resource = resource
    self._function = function
    self._partitions = partitions
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
        self.spec, self._resource, self._dataset_function, self._partitions)

  #=============================================================================
  # Indexing & Slicing
  #=============================================================================
  def __getitem__(self, key):
    """Returns the specified piece of this IOTensor."""
    # Find out the indices based on length and key,
    # based on python slice()'s indices method:
    index = key if isinstance(key, slice) else slice(key, key + 1)
    start, stop, _ = index.indices(self.shape[0])
    if start >= self.shape[0]:
      raise IndexError("index %s is out of range" % key)
    item = self._function(self._resource, start=start, stop=stop)
    return tf.squeeze(item, axis=[0]) if (stop == start + 1) else item

  def __len__(self):
    """Returns the total number of items of this IOTensor."""
    return self.shape[0]


  #=============================================================================
  # Windowing
  #=============================================================================
  def window(self, size):
    """Returns the sliding window of this IOTensor."""
    spec = tf.TensorSpec(
        tf.TensorShape(self.shape.dims[0] - size + 1).concatenate(size),
        self.dtype)
    class _Function(object):
      def __init__(self, func, spec, size):
        self._func = func
        self._spec = spec
        self._size = size
      def __call__(self, resource, start, stop):
        return tf.reshape(
            tf.image.extract_patches(
                tf.reshape(
                    self._func(
                        resource,
                        start, stop + self._size - 1),
                    [1, 1, stop + self._size - 1 - start, 1]),
                sizes=[1, 1, self._size, 1],
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'),
            self._spec.shape)
    return BaseIOTensor(spec,
                        self._resource,
                        _Function(self._function, spec, size),
                        partitions=None, internal=True)

  #=============================================================================
  # Tensor Type Conversions
  #=============================================================================

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

class TensorIOTensor(BaseIOTensor):
  """TensorIOTensor

  A `TensorIOTensor` is an `IOTensor` from a regular `Tensor`.
  """

  def __init__(self,
               tensor,
               internal=False):
    tensor = tf.convert_to_tensor(tensor)

    class _Function(object):
      """_Function"""
      def __init__(self, tensor):
        self._base_start = [0 for _ in tensor.shape.as_list()]
        self._base_size = [-1 for _ in tensor.shape.as_list()]

      def __call__(self, resource, start, stop):
        slice_start = self._base_start
        slice_size = self._base_size
        slice_start[0] = start
        slice_size[0] = stop - start
        return tf.slice(resource, slice_start, slice_size)

    self._tensor = tensor

    super(TensorIOTensor, self).__init__(
        tf.TensorSpec(tensor.shape, tensor.dtype),
        tensor, _Function(tensor), None, internal=internal)

  #=============================================================================
  # Tensor Type Conversions
  #=============================================================================

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
      return self._tensor

class _TableIOTensor(_IOTensor):
  """_TableIOTensor"""

  def __init__(self,
               spec,
               columns,
               resource,
               function,
               partitions,
               internal=False):
    self._columns = columns
    self._resource = resource
    self._function = function
    self._partitions = partitions
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
    column_index = self.columns.index(
        next(e for e in self.columns if e == column))
    spec = tf.nest.flatten(self.spec)[column_index]
    class _Function(object):
      def __init__(self, func, spec, column):
        self._func = func
        self._shape = tf.TensorShape([None]).concatenate(spec.shape[1:])
        self._dtype = spec.dtype
        self._component = column

      def __call__(self, resource, start, stop):
        return self._func(
            resource, start=start, stop=stop,
            component=self._component,
            shape=self._shape, dtype=self._dtype)
    function = _Function(self._function, spec, column)
    return BaseIOTensor(
        spec, self._resource, function, self._partitions,
        internal=True)

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
    class _Function(object):
      """_Function"""
      def __init__(self, func, spec, columns):
        self._func = func
        self._spec = [tf.TensorSpec(
            tf.TensorShape([None]).concatenate(e.shape[1:]),
            e.dtype) for e in spec]
        self._columns = columns
        self._components = zip(
            tf.nest.flatten(self._spec), tf.nest.flatten(self._columns))

      def __call__(self, resource, start, stop):
        return tf.nest.pack_sequence_as(
            self._columns,
            [self._func(
                resource, start, stop,
                component=component,
                shape=e.shape,
                dtype=e.dtype) for (e, component) in self._components])

    return _IOTensorDataset(
        self.spec, self._resource,
        _Function(self._function, self.spec, self.columns), self._partitions)


class _CollectionIOTensor(_IOTensor):
  """_CollectionIOTensor

  `CollectionIOTensor` is differnt from `TableIOTensor` in that each
  component could have different shapes. While additional table-wide
  operations are planned to be supported for `TableIOTensor` so that
  the same operations could be applied to every column, there is no plan
  to support the same in `CollectionIOTensor`. In other words,
  `CollectionIOTensor` is only a dictionary with values consisting
  of `BaseIOTensor`.
  """

  def __init__(self,
               spec,
               keys,
               values,
               internal=False):
    self._keys = keys
    self._values = values
    super(_CollectionIOTensor, self).__init__(
        spec, internal=internal)

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def keys(self):
    """The names of columns"""
    return self._keys

  def __call__(self, key):
    """Return a BaseIOTensor with key named `key`"""
    key_index = self.keys.index(
        next(e for e in self.keys if e == key))
    return self._values[key_index]

class _SeriesIOTensor(_IOTensor):
  """_SeriesIOTensor"""

  def __init__(self,
               spec,
               resource,
               function,
               internal=False):
    self._resource = resource

    class _SeriesFunction(object):
      def __init__(self, func, component, spec):
        self._func = func
        self._component = component
        self._shape = spec.shape
        self._dtype = spec.dtype
      def __call__(self, resource, start, stop):
        return self._func(resource, start, stop,
                          component=self._component,
                          shape=self._shape, dtype=self._dtype)

    self._index_function = _SeriesFunction(function, "index", spec[0])
    self._value_function = _SeriesFunction(function, "value", spec[1])
    super(_SeriesIOTensor, self).__init__(
        spec, internal=internal)

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def index(self):
    """The index column of the series"""
    return BaseIOTensor(
        self.spec[0],
        self._resource,
        self._index_function,
        partitions=None, internal=True)

  @property
  def value(self):
    """The value column of the series"""
    return BaseIOTensor(
        self.spec[1],
        self._resource,
        self._value_function,
        partitions=None, internal=True)

class _KeyValueIOTensorDataset(tf.compat.v2.data.Dataset):
  """_KeyValueIOTensorDataset"""

  def __init__(self,
               iterable_init, iterable_next,
               mapping_resource, mapping_function):
    with tf.name_scope("IterableIOTensorDataset"):
      resource = iterable_init()

      capacity = 4096
      dataset = tf.compat.v2.data.Dataset.range(0, sys.maxsize, capacity)
      def func(_):
        k = iterable_next(resource, capacity)
        v = mapping_function(mapping_resource, k)
        return (k, v)
      dataset = dataset.map(func)
      dataset = dataset.apply(
          tf.data.experimental.take_while(
              lambda k, v: tf.greater(tf.shape(k)[0], 0)))
      dataset = dataset.unbatch()

      self._iterable_init = iterable_init
      self._iterable_next = iterable_next

      self._mapping_resource = mapping_resource
      self._mapping_function = mapping_function

      self._resource = resource
      self._dataset = dataset
      super(_KeyValueIOTensorDataset, self).__init__(
          self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec

class _KeyValueIOTensor(_IOTensor):
  """_KeyValueIOTensor"""

  def __init__(self,
               spec,
               resource,
               function,
               iterable_init,
               iterable_next,
               internal=False):
    self._resource = resource
    self._function = function
    self._iterable_init = iterable_init
    self._iterable_next = iterable_next
    super(_KeyValueIOTensor, self).__init__(
        spec, internal=internal)

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
    return _KeyValueIOTensorDataset(
        self._iterable_init, self._iterable_next,
        self._resource, self._function)

  #=============================================================================
  # Iterator
  #=============================================================================
  def __iter__(self):
    with tf.name_scope("KeyValueIOTensorIter"):
      resource = self._iterable_init()
      capacity = 1
      while True:
        value = self._iterable_next(resource, capacity)
        if tf.shape(value)[0].numpy() < capacity:
          return
        yield value[0]

  #=============================================================================
  # Indexing
  #=============================================================================
  def __getitem__(self, key):
    """Returns the specified piece of this IOTensor."""
    return self._function(self._resource, key)
