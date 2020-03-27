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
"""CSVIOTensor"""

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops


class _IOTensorComponentLabelFunction:
    """_IOTensorComponentLabelFunction"""

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
            filter=["label"],
            shape=self._shape,
            dtype=self._dtype,
        )

    @property
    def length(self):
        return self._length


class CSVIOTensor(io_tensor_ops._TableIOTensor):  # pylint: disable=protected-access
    """CSVIOTensor"""

    # =============================================================================
    # Constructor (private)
    # =============================================================================
    def __init__(self, filename, internal=False):
        with tf.name_scope("CSVIOTensor") as scope:
            resource, columns = core_ops.io_csv_readable_init(
                filename,
                container=scope,
                shared_name="{}/{}".format(filename, uuid.uuid4().hex),
            )
            columns = [column.decode() for column in columns.numpy().tolist()]
            elements = []
            for column in columns:
                shape, dtype = core_ops.io_csv_readable_spec(resource, column)
                shape = tf.TensorShape(shape.numpy())
                dtype = tf.as_dtype(dtype.numpy())
                spec = tf.TensorSpec(shape, dtype, column)
                function = io_tensor_ops._IOTensorComponentFunction(  # pylint: disable=protected-access
                    core_ops.io_csv_readable_read, resource, column, shape, dtype
                )
                elements.append(
                    io_tensor_ops.BaseIOTensor(spec, function, internal=internal)
                )
            spec = tuple([e.spec for e in elements])

            self._resource = resource
            super().__init__(spec, columns, elements, internal=internal)

    # =============================================================================
    # IsNull checking
    # =============================================================================
    def isnull(self, column):
        """Return a BaseIOTensor of bool for null values in `column`"""
        column_index = self.columns.index(next(e for e in self.columns if e == column))
        spec = tf.nest.flatten(self.spec)[column_index]
        # change spec to bool
        spec = tf.TensorSpec(spec.shape, tf.bool)
        function = _IOTensorComponentLabelFunction(
            core_ops.io_csv_readable_read,
            self._resource,
            column,
            spec.shape,
            spec.dtype,
        )

        return io_tensor_ops.BaseIOTensor(spec, function, internal=True)
