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
"""ParquetDataset"""

import collections

import tensorflow as tf
from tensorflow_io.python.ops import core_ops


class ParquetIODataset(tf.data.Dataset):
    """ParquetIODataset"""

    def __init__(self, filename, columns=None, internal=True):
        """ParquetIODataset."""
        assert internal
        with tf.name_scope("ParquetIODataset"):
            components, shapes, dtypes = core_ops.io_parquet_readable_info(
                filename, shared=filename, container="ParquetIODataset"
            )

            def component_f(components, column):
                component = tf.boolean_mask(
                    components, tf.math.equal(components, column)
                )[0]
                return component

            def shape_f(shapes, components, column):
                shape = tf.boolean_mask(shapes, tf.math.equal(components, column))[0]
                shape = tf.boolean_mask(shape, tf.math.greater_equal(shape, 0))
                return shape

            def dtype_f(dtypes, components, column):
                dtype = tf.boolean_mask(dtypes, tf.math.equal(components, column))[0]
                dtype = tf.as_dtype(dtype.numpy())
                return dtype

            if not tf.executing_eagerly():
                if columns is None or not isinstance(columns, dict):
                    raise ValueError(
                        "The `columns` parameter can only be "
                        "a dictionary in graph execution, mapping "
                        "feature names to `tf.TensorSpec`."
                    )
                shapes = [shape_f(shapes, components, column) for column in columns]
                dtypes = [
                    spec if isinstance(spec, tf.dtypes.DType) else spec.dtype
                    for column, spec in columns.items()
                ]
                components = [component_f(components, column) for column in columns]
                column_names = list(columns.keys())
            elif columns is not None:
                shapes = [shape_f(shapes, components, column) for column in columns]
                dtypes = [dtype_f(dtypes, components, column) for column in columns]
                components = (
                    list(columns.keys()) if isinstance(columns, dict) else columns
                )
                column_names = components
            else:
                shapes = tf.unstack(shapes)
                dtypes = [tf.as_dtype(dtype.numpy()) for dtype in tf.unstack(dtypes)]
                components = [component.numpy() for component in tf.unstack(components)]
                column_names = components

            self._filename = filename
            self._components = components
            self._shapes = shapes
            self._dtypes = dtypes

            def dataset_f(component, shape, dtype):
                step = 4096
                indices_start = tf.data.Dataset.range(0, shape[0], step)
                indices_stop = indices_start.skip(1).concatenate(
                    tf.data.Dataset.from_tensor_slices(
                        tf.convert_to_tensor([shape[0]], tf.int64)
                    )
                )
                dataset = tf.data.Dataset.zip((indices_start, indices_stop))

                def f(start, stop):
                    return core_ops.io_parquet_readable_read(
                        input=self._filename,
                        shared=self._filename,
                        component=component,
                        shape=shape,
                        start=start,
                        stop=stop,
                        dtype=dtype,
                        container="ParquetIODataset",
                    )

                dataset = dataset.map(f)
                dataset = dataset.unbatch()
                return dataset

            entries = list(zip(components, shapes, dtypes))
            datasets = [
                dataset_f(component, shape, dtype)
                for component, shape, dtype in entries
            ]
            self._dataset = tf.data.Dataset.zip(
                collections.OrderedDict(list(zip(column_names, datasets)))
            )

            # Override the default `element_spec` with given specs if available.
            if isinstance(columns, dict) and all(
                isinstance(val, tf.TensorSpec) for val in columns.values()
            ):
                self._element_spec = collections.OrderedDict(columns)
            else:
                self._element_spec = None

            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        if self._element_spec:
            return self._element_spec
        return self._dataset.element_spec
