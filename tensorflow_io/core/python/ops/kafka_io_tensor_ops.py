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
"""KafkaIOTensor"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class KafkaIOTensor:
    """KafkaIOTensor"""

    # =============================================================================
    # Constructor (private)
    # =============================================================================
    def __init__(self, topic, partition, servers, configuration, internal=True):
        with tf.name_scope("KafkaIOTensor"):
            assert internal

            metadata = list(configuration or [])
            if servers is not None:
                metadata.append("bootstrap.servers=%s" % servers)
            resource = core_ops.io_kafka_readable_init(
                topic, partition, offset=0, metadata=metadata
            )

            self._resource = resource
            super().__init__()

    # =============================================================================
    # Accessors
    # =============================================================================

    @property
    def shape(self):
        """Returns the `TensorShape` that represents the shape of the tensor."""
        return tf.TensorShape([None])

    @property
    def dtype(self):
        """Returns the `dtype` of elements in the tensor."""
        return tf.string

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
        item, _ = core_ops.io_kafka_readable_read(self._resource, start=0, stop=-1)
        return item

    # =============================================================================
    # Indexing and slicing
    # =============================================================================
    def __getitem__(self, key):
        """Returns the specified piece of this IOTensor."""
        if isinstance(key, slice):
            item, _ = core_ops.io_kafka_readable_read(
                self._resource, key.start, key.stop
            )
            return item
        item, _ = core_ops.io_kafka_readable_read(self._resource, key, key + 1)
        if tf.shape(item)[0] == 0:
            raise IndexError("index %s is out of range" % key)
        return item[0]
