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
"""HDF5IOTensor"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.ops import io_tensor_ops


class BaseHDF5GraphIOTensor:
    """BaseHDF5GraphIOTensor"""

    # =============================================================================
    # Constructor (private)
    # =============================================================================
    def __init__(self, filename, component, shape, dtype, internal=False):
        with tf.name_scope("BaseHDF5GraphIOTensor"):
            assert internal
            self._filename = filename
            self._component = component
            self._shape = shape
            self._dtype = dtype
            super().__init__()

    # =============================================================================
    # Accessors
    # =============================================================================

    @property
    def shape(self):
        """Returns the `TensorShape` that represents the shape of the tensor."""
        return self._shape

    @property
    def dtype(self):
        """Returns the `dtype` of elements in the tensor."""
        return self._dtype

    # =============================================================================
    # String Encoding
    # =============================================================================
    def __repr__(self):
        return "<{}: shape={}, dtype={}>".format(
            self.__class__.__name__, self.shape, self.dtype
        )

    # =============================================================================
    # Tensor Type Conversions
    # =============================================================================

    def to_tensor(self):
        """Converts this `IOTensor` into a `tf.Tensor`.

        Args:
            name: A name prefix for the returned tensors (optional).

        Returns:
            A `Tensor` with value obtained from this `IOTensor`.
        """
        return core_ops.io_hdf5_readable_read(
            input=self._filename,
            shared=self._filename,
            component=self._component,
            shape=self._shape,
            start=0,
            stop=-1,
            dtype=self._dtype,
            container="HDF5IOTensor",
        )

    # =============================================================================
    # Indexing and slicing
    # =============================================================================
    def __getitem__(self, key):
        """Returns the specified piece of this IOTensor."""
        # always convert to tuple to process
        if not isinstance(key, tuple):
            key = tuple([key])
        # get the start and stop of each element
        indices = [
            (k.start, k.stop) if isinstance(k, slice) else (k, k + 1) for k in key
        ]
        # get the start and stop, and use 0 (start) and -1 (stop) if needed
        indices = list(zip(*indices))
        start = [0 if e is None else e for e in indices[0]]
        stop = [-1 if e is None else e for e in indices[1]]

        item = core_ops.io_hdf5_readable_read(
            input=self._filename,
            shared=self._filename,
            component=self._component,
            shape=self._shape,
            start=start,
            stop=stop,
            dtype=self._dtype,
            container="HDF5IOTensor",
        )

        # in case certain dimension is not slice, then this dimension will need to
        # collapse as `0`, otherwise `:` or `slice(None, None, None)`
        indices = [slice(None) if isinstance(k, slice) else 0 for k in key]

        return item.__getitem__(indices)

    def __len__(self):
        """Returns the total number of items of this IOTensor."""
        return self._shape[0]


class HDF5IOTensor(
    io_tensor_ops._CollectionIOTensor
):  # pylint: disable=protected-access
    """HDF5IOTensor"""

    # =============================================================================
    # Constructor (private)
    # =============================================================================
    def __init__(self, filename, spec=None, internal=False):
        with tf.name_scope("HDF5IOTensor"):
            columns, shapes, dtypes = core_ops.io_hdf5_readable_info(
                filename, shared=filename, container="HDF5IOTensor"
            )
            if tf.executing_eagerly():
                columns = tf.unstack(columns)
                shapes = [
                    tf.boolean_mask(shape, tf.math.greater_equal(shape, 0))
                    for shape in tf.unstack(shapes)
                ]
                dtypes = [tf.as_dtype(dtype.numpy()) for dtype in tf.unstack(dtypes)]
                entries = [
                    tf.TensorSpec(shape, dtype, column)
                    for (shape, dtype, column) in zip(shapes, dtypes, columns)
                ]
            else:
                assert spec is not None

                entries = spec.items()

                def f(column, columns, shapes):
                    shape = tf.boolean_mask(shapes, tf.math.equal(columns, column))[0]
                    shape = tf.boolean_mask(shape, tf.math.greater_equal(shape, 0))
                    return shape

                shapes = [f(column, columns, shapes) for column, _ in entries]
                dtypes = [
                    entry if isinstance(entry, tf.dtypes.DType) else entry.dtype
                    for _, entry in entries
                ]
                columns = [column for column, _ in entries]

                entries = [
                    tf.TensorSpec(None, dtype, column)
                    for (dtype, column) in zip(dtypes, columns)
                ]

            def g(entry, shape):
                return BaseHDF5GraphIOTensor(
                    filename, entry.name, shape, entry.dtype, internal=True
                )

            elements = [g(entry, shape) for (entry, shape) in zip(entries, shapes)]
            spec = tuple(entries)
            super().__init__(spec, columns, elements, internal=internal)
