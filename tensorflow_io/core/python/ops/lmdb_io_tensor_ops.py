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
"""LMDBIOTensor"""

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops


class _IOTensorMappingFunction:
    def __init__(self, function, resource):
        self._function = function
        self._resource = resource

    def __call__(self, key):
        return self._function(self._resource, key)


class LMDBIOTensor(io_tensor_ops._KeyValueIOTensor):  # pylint: disable=protected-access
    """LMDBIOTensor"""

    # =============================================================================
    # Constructor (private)
    # =============================================================================
    def __init__(self, filename, internal=False):
        with tf.name_scope("LMDBIOTensor") as scope:
            resource = core_ops.io_lmdb_mapping_init(
                filename,
                container=scope,
                shared_name="{}/{}".format(filename, uuid.uuid4().hex),
            )
            spec = (
                tf.TensorSpec(tf.TensorShape([None]), tf.string),
                tf.TensorSpec(tf.TensorShape([None]), tf.string),
            )

            class _IterableInit:
                def __init__(self, func, filename):
                    self._func = func
                    self._filename = filename

                def __call__(self):
                    with tf.name_scope("IterableInit") as scope:
                        return self._func(
                            self._filename,
                            container=scope,
                            shared_name="{}/{}".format(
                                self._filename, uuid.uuid4().hex
                            ),
                        )

            class _IterableNext:
                def __init__(self, func, shape, dtype):
                    self._func = func
                    self._shape = shape
                    self._dtype = dtype
                    self._index = 0

                def __call__(self, resource):
                    return self._func(
                        resource,
                        start=self._index,
                        stop=self._index + 1,
                        shape=self._shape,
                        dtype=self._dtype,
                    )

            super().__init__(
                spec,
                _IOTensorMappingFunction(core_ops.io_lmdb_mapping_read, resource),
                _IterableInit(core_ops.io_lmdb_readable_init, filename),
                _IterableNext(
                    core_ops.io_lmdb_readable_read, tf.TensorShape([None]), tf.string
                ),
                internal=internal,
            )
