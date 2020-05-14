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
"""HDF5Dataset"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class HDF5IODataset(tf.data.Dataset):
    """HDF5IODataset"""

    def __init__(self, filename, dataset, spec=None, internal=True):
        """HDF5IODataset."""
        with tf.name_scope("HDF5IODataset"):
            assert internal

            components, shapes, dtypes = core_ops.io_hdf5_readable_info(
                filename, shared=filename, container="HDF5IODataset"
            )
            shape = tf.boolean_mask(shapes, tf.math.equal(components, dataset))[0]
            shape = tf.boolean_mask(shape, tf.math.greater_equal(shape, 0))

            if tf.executing_eagerly():
                dtype = tf.boolean_mask(dtypes, tf.math.equal(components, dataset))[0]
                dtype = tf.as_dtype(dtype.numpy())
            else:
                assert spec is not None
                dtype = spec if isinstance(spec, tf.dtypes.DType) else spec.dtype

            self._filename = filename
            self._component = dataset
            self._shape = shape
            self._dtype = dtype

            step = 1024
            indices_start = tf.data.Dataset.range(0, shape[0], step)
            indices_stop = indices_start.skip(1).concatenate(
                tf.data.Dataset.from_tensor_slices([shape[0]])
            )
            dataset = tf.data.Dataset.zip((indices_start, indices_stop))

            def f(start, stop):
                return core_ops.io_hdf5_readable_read(
                    input=self._filename,
                    shared=self._filename,
                    component=self._component,
                    shape=self._shape,
                    start=start,
                    stop=stop,
                    dtype=self._dtype,
                    container="HDF5IODataset",
                )

            dataset = dataset.map(f)
            dataset = dataset.unbatch()

            self._dataset = dataset
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
