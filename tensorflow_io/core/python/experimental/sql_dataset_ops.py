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
"""SQLDataset"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class SQLIODataset(tf.data.Dataset):
    """SQLIODataset"""

    def __init__(self, query, endpoint=None, spec=None, internal=True):
        """SQLIODataset."""
        with tf.name_scope("SQLIODataset"):
            assert internal
            endpoint = endpoint or ""
            resource, count, fields, dtypes = core_ops.io_sql_iterable_init(
                query, endpoint
            )
            if spec is None:
                fields = tf.unstack(fields)
                dtypes = tf.unstack(dtypes)
                spec = {
                    field.numpy().decode(): tf.TensorSpec(
                        [None], tf.as_dtype(dtype.numpy()), field.numpy().decode()
                    )
                    for (field, dtype) in zip(fields, dtypes)
                }
            else:
                # Make sure shape is [None] and name is part of the spec
                spec = {k: tf.TensorSpec([None], v.dtype, k) for k, v in spec.items()}

            flatten = tf.nest.flatten(spec)
            fields = [e.name for e in flatten]
            dtypes = [e.dtype for e in flatten]

            self._resource = resource
            dataset = tf.data.Dataset.range(0, count)

            def f(index):
                return tf.nest.pack_sequence_as(
                    spec, core_ops.io_sql_iterable_read(resource, index, fields, dtypes)
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
