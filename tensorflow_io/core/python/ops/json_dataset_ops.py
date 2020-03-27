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
"""JSONDataset"""

import sys
import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class _JSONIODatasetFunction:
    def __init__(self, function, resource, component, shape, dtype):
        self._function = function
        self._resource = resource
        self._component = component
        self._shape = tf.TensorShape([None]).concatenate(shape[1:])
        self._dtype = dtype

    def __call__(self, start, stop):
        return self._function(
            self._resource,
            start=start,
            stop=stop,
            component=self._component,
            shape=self._shape,
            dtype=self._dtype,
        )


class JSONIODataset(tf.compat.v2.data.Dataset):
    """JSONIODataset"""

    def __init__(self, filename, columns=None, mode=None, internal=True):
        """JSONIODataset."""
        if not internal:
            raise ValueError(
                "JSONIODataset constructor is private; please use one "
                "of the factory methods instead (e.g., "
                "IODataset.from_json())"
            )
        with tf.name_scope("JSONIODataset") as scope:
            capacity = 4096

            metadata = [] if mode is None else ["mode: %s" % mode]
            resource, columns_v = core_ops.io_json_readable_init(
                filename,
                metadata=metadata,
                container=scope,
                shared_name="{}/{}".format(filename, uuid.uuid4().hex),
            )
            columns = columns if columns is not None else columns_v.numpy()

            columns_dataset = []

            columns_function = []
            for column in columns:
                shape, dtype = core_ops.io_json_readable_spec(resource, column)
                shape = tf.TensorShape([None if e < 0 else e for e in shape.numpy()])
                dtype = tf.as_dtype(dtype.numpy())
                function = _JSONIODatasetFunction(
                    core_ops.io_json_readable_read, resource, column, shape, dtype
                )
                columns_function.append(function)

            for (column, function) in zip(columns, columns_function):
                column_dataset = tf.compat.v2.data.Dataset.range(
                    0, sys.maxsize, capacity
                )
                column_dataset = column_dataset.map(
                    lambda index: function(index, index + capacity)
                )
                column_dataset = column_dataset.apply(
                    tf.data.experimental.take_while(
                        lambda v: tf.greater(tf.shape(v)[0], 0)
                    )
                )
                columns_dataset.append(column_dataset)
            if len(columns_dataset) == 1:
                dataset = columns_dataset[0]
            else:
                dataset = tf.compat.v2.data.Dataset.zip(tuple(columns_dataset))
            dataset = dataset.unbatch()

            self._function = columns_function
            self._dataset = dataset
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
