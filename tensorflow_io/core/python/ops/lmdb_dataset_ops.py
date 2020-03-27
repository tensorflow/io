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
"""LMDBDataset"""

import sys
import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class _LMDBIODatasetFunction:
    def __init__(self, resource):
        self._resource = resource

    def __call__(self, start, stop):
        return core_ops.io_kafka_readable_read(
            self._resource,
            start=start,
            stop=stop,
            shape=tf.TensorShape([None]),
            dtype=tf.string,
        )


class LMDBIODataset(tf.compat.v2.data.Dataset):
    """LMDBIODataset"""

    def __init__(self, filename, **kwargs):
        with tf.name_scope("LMDBIODataset") as scope:
            mapping = core_ops.io_lmdb_mapping_init(
                filename,
                container=scope,
                shared_name="{}/{}".format(filename, uuid.uuid4().hex),
            )
            resource = core_ops.io_lmdb_readable_init(
                filename,
                container=scope,
                shared_name="{}/{}".format(filename, uuid.uuid4().hex),
            )
            capacity = kwargs.get("capacity", 4096)
            dataset = tf.compat.v2.data.Dataset.range(0, sys.maxsize, capacity)
            dataset = dataset.map(
                lambda index: core_ops.io_lmdb_readable_read(
                    resource,
                    start=index,
                    stop=index + capacity,
                    shape=tf.TensorShape([None]),
                    dtype=tf.string,
                )
            )
            dataset = dataset.apply(
                tf.data.experimental.take_while(lambda v: tf.greater(tf.shape(v)[0], 0))
            )
            dataset = dataset.map(
                lambda key: (key, core_ops.io_lmdb_mapping_read(mapping, key))
            )
            dataset = dataset.unbatch()

            self._mapping = mapping
            self._resource = resource
            self._capacity = capacity
            self._dataset = dataset
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
