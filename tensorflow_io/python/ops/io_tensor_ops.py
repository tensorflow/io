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

import tensorflow as tf


class _IOTensorMeta(property):
    """_IOTensorMeta is a decorator that is viewable to __repr__"""


class _IOTensorComponentFunction:
    """_IOTensorComponentFunction will translate call"""

    def __init__(self, function, resource, component, shape, dtype):
        self._function = function
        self._resource = resource
        self._component = component
        self._length = shape[0]
        self._shape = tf.TensorShape([None]).concatenate(shape[1:])
        self._dtype = dtype

    def __call__(self, start, stop):
        start, stop, _ = slice(start, stop).indices(self._length)
        return self._function(
            self._resource,
            start=start,
            stop=stop,
            component=self._component,
            shape=self._shape,
            dtype=self._dtype,
        )

    @property
    def length(self):
        return self._length


class _IOTensorIterablePartitionedFunction:
    """PartitionedFunction will translate call to cached Function call"""

    # function: next call of the iterable
    def __init__(self, function, shape):
        self._function = function
        self._partitions = []
        self._length = None
        self._slice_suffix_start = [0 for _ in shape[1:]]
        self._slice_suffix_size = list(shape[1:])

    def __call__(self, start, stop):
        while self._length is None:
            # if stop is not None then resolved partitions have to cover stop
            # if stop is None then all partitions has to be resolved
            if stop is not None:
                if stop <= sum([e.shape[0] for e in self._partitions]):
                    break
            # resolve one step
            partition = self._function()
            if partition.shape[0] == 0:
                self._length = sum([e.shape[0] for e in self._partitions])
            else:
                self._partitions.append(partition)
            partitions_indices = (
                tf.cumsum([e.shape[0] for e in self._partitions]).numpy().tolist()
            )
            self._partitions_start = list([0] + partitions_indices[:-1])
            self._partitions_stop = partitions_indices
        length = self._partitions_stop[-1]
        index = slice(start, stop)
        start, stop, _ = index.indices(length)
        if start >= length:
            raise IndexError("index %s is out of range" % index)
        indices_start = tf.math.maximum(self._partitions_start, start)
        indices_stop = tf.math.minimum(self._partitions_stop, stop)
        indices_hit = tf.math.less(indices_start, indices_stop)
        indices = tf.squeeze(tf.compat.v2.where(indices_hit), [1])
        return self._partitions_read(indices_start, indices_stop, indices)

    @property
    def length(self):
        """length"""
        while self._length is None:
            # resolve until length is available
            partition = self._function()
            if partition.shape[0] == 0:
                self._length = sum([e.shape[0] for e in self._partitions])
            else:
                self._partitions.append(partition)
            partitions_indices = (
                tf.cumsum([e.shape[0] for e in self._partitions]).numpy().tolist()
            )
            self._partitions_start = list([0] + partitions_indices[:-1])
            self._partitions_stop = partitions_indices
        return self._length

    def _partitions_read(self, indices_start, indices_stop, indices):
        """_partitions_read"""
        items = []
        # TODO: change to tf.while_loop
        for index in indices:
            slice_start = indices_start[index] - self._partitions_start[index]
            slice_size = indices_stop[index] - indices_start[index]
            slice_start = [slice_start] + self._slice_suffix_start
            slice_size = [slice_size] + self._slice_suffix_size
            item = tf.slice(self._partitions[index], slice_start, slice_size)
            items.append(item)
        return tf.concat(items, axis=0)


class _IOTensorPartitionedFunction:
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
                    self._partitions_stop[index],
                )
            slice_start = indices_start[index] - self._partitions_start[index]
            slice_size = indices_stop[index] - indices_start[index]
            item = tf.slice(self._tensors[index], [slice_start], [slice_size])
            items.append(item)
        return tf.concat(items, axis=0)


class _IOTensor:
    """_IOTensor"""

    def __init__(self, spec, internal=False):
        if not internal:
            raise ValueError(
                "IOTensor constructor is private; please use one "
                "of the factory methods instead (e.g., "
                "IOTensor.from_tensor())"
            )
        self._spec = spec
        super().__init__()

    # =============================================================================
    # Accessors
    # =============================================================================

    @property
    def spec(self):
        """The `TensorSpec` of values in this tensor."""
        return self._spec

    # =============================================================================
    # String Encoding
    # =============================================================================
    def __repr__(self):
        meta = "".join(
            [
                ", {}={}".format(k, repr(v.__get__(self)))
                for k, v in self.__class__.__dict__.items()
                if isinstance(v, _IOTensorMeta)
            ]
        )
        return "<{}: spec={}{}>".format(self.__class__.__name__, self.spec, meta)


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

    def __init__(self, spec, function, internal=False):
        # function used for dataset should not be partitioned.
        self._function = function
        super().__init__(spec, internal=internal)

    # =============================================================================
    # Accessors
    # =============================================================================

    @property
    def shape(self):
        """Returns the `TensorShape` that represents the shape of the tensor."""
        return self.spec.shape

    @property
    def dtype(self):
        """Returns the `dtype` of elements in the tensor."""
        return self.spec.dtype

    # =============================================================================
    # Indexing & Slicing
    # =============================================================================
    def __getitem__(self, key):
        """Returns the specified piece of this IOTensor."""
        # Find out the indices based on length and key,
        # based on python slice()'s indices method:
        index = key if isinstance(key, slice) else slice(key, key + 1)
        items = self._function(start=index.start, stop=index.stop)
        return tf.squeeze(items, axis=[0]) if items.shape[0] == 1 else items

    def __len__(self):
        """Returns the total number of items of this IOTensor."""
        return self._function.length

    # =============================================================================
    # Windowing
    # =============================================================================
    def window(self, size):
        """Returns the sliding window of this IOTensor."""
        spec = tf.TensorSpec(
            tf.TensorShape(self._function.length - size + 1).concatenate(size),
            self.dtype,
        )

        class _Function:
            """_Function"""

            def __init__(self, func, spec, size):
                self._func = func
                self._spec = spec
                self._size = size
                self._length = self._spec.shape[0]

            def __call__(self, start, stop):
                start, stop, _ = slice(start, stop).indices(self._length)
                if start >= self._length:
                    raise IndexError("index %s is out of range" % slice(start, stop))
                return tf.reshape(
                    tf.image.extract_patches(
                        tf.reshape(
                            self._func(start, stop + self._size - 1),
                            [1, 1, stop + self._size - 1 - start, 1],
                        ),
                        sizes=[1, 1, self._size, 1],
                        strides=[1, 1, 1, 1],
                        rates=[1, 1, 1, 1],
                        padding="VALID",
                    ),
                    self._spec.shape,
                )

        return BaseIOTensor(spec, _Function(self._function, spec, size), internal=True)

    # =============================================================================
    # Tensor Type Conversions
    # =============================================================================

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


class ScalarIOTensor(BaseIOTensor):
    """ScalarIOTensor

    A `ScalarIOTensor` is an `IOTensor` from a scalar `Tensor`.
    """

    def __init__(self, spec, tensor, internal=False):
        tensor = tf.convert_to_tensor(tensor)

        self._tensor = tensor
        super().__init__(spec, None, internal=internal)

    # =============================================================================
    # Tensor Type Conversions
    # =============================================================================

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


class TensorIOTensor(BaseIOTensor):
    """TensorIOTensor

    A `TensorIOTensor` is an `IOTensor` from a regular `Tensor`.
    """

    def __init__(self, tensor, internal=False):
        tensor = tf.convert_to_tensor(tensor)

        class _Function:
            """_Function"""

            def __init__(self, tensor):
                self._tensor = tensor
                self._base_start = [0 for _ in tensor.shape.as_list()]
                self._base_size = [-1 for _ in tensor.shape.as_list()]
                self._length = tensor.shape[0]

            def __call__(self, start, stop):
                start, stop, _ = slice(start, stop).indices(self._length)
                if start >= self._length:
                    raise IndexError("index %s is out of range" % slice(start, stop))
                slice_start = self._base_start
                slice_size = self._base_size
                slice_start[0] = start
                slice_size[0] = stop - start
                return tf.slice(self._tensor, slice_start, slice_size)

            @property
            def length(self):
                return self._length

        self._tensor = tensor

        super().__init__(
            tf.TensorSpec(tensor.shape, tensor.dtype),
            _Function(tensor),
            internal=internal,
        )

    # =============================================================================
    # Tensor Type Conversions
    # =============================================================================

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

    def __init__(self, spec, columns, values, internal=False):
        self._columns = columns
        self._values = values
        super().__init__(spec, internal=internal)

    # =============================================================================
    # Accessors
    # =============================================================================

    @property
    def columns(self):
        """The names of columns"""
        return self._columns

    def __call__(self, column):
        """Return a BaseIOTensor with column named `column`"""
        column_index = self.columns.index(next(e for e in self.columns if e == column))
        return self._values[column_index]


class _CollectionIOTensor(_IOTensor):
    """_CollectionIOTensor

    `CollectionIOTensor` is different from `TableIOTensor` in that each
    component could have different shapes. While additional table-wide
    operations are planned to be supported for `TableIOTensor` so that
    the same operations could be applied to every column, there is no plan
    to support the same in `CollectionIOTensor`. In other words,
    `CollectionIOTensor` is only a dictionary with values consisting
    of `BaseIOTensor`.
    """

    def __init__(self, spec, keys, values, internal=False):
        self._keys = keys
        self._values = values
        super().__init__(spec, internal=internal)

    # =============================================================================
    # Accessors
    # =============================================================================

    @property
    def keys(self):
        """The names of columns"""
        return self._keys

    def __call__(self, key):
        """Return a BaseIOTensor with key named `key`"""
        key_index = self.keys.index(next(e for e in self.keys if e == key))
        return self._values[key_index]


class _SeriesIOTensor(_IOTensor):
    """_SeriesIOTensor"""

    def __init__(self, spec, index, value, internal=False):
        self._index = index
        self._value = value
        super().__init__(spec, internal=internal)

    # =============================================================================
    # Accessors
    # =============================================================================

    @property
    def index(self):
        """The index column of the series"""
        return self._index

    @property
    def value(self):
        """The value column of the series"""
        return self._value


class _KeyValueIOTensor(_IOTensor):
    """_KeyValueIOTensor"""

    def __init__(self, spec, function, iterable_init, iterable_next, internal=False):
        self._function = function
        self._iterable_init = iterable_init
        self._iterable_next = iterable_next
        super().__init__(spec, internal=internal)

    # =============================================================================
    # Iterator
    # =============================================================================
    def __iter__(self):
        with tf.name_scope("KeyValueIOTensorIter"):
            resource = self._iterable_init()
            while True:
                value = self._iterable_next(resource)
                if tf.shape(value)[0].numpy() == 0:
                    return
                yield value[0]

    # =============================================================================
    # Indexing
    # =============================================================================
    def __getitem__(self, key):
        """Returns the specified piece of this IOTensor."""
        return self._function(key)
