# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""GRPCStreamIODataset."""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class GRPCStreamIODataset(tf.data.Dataset):
    """GRPCStreamIODataset"""

    def __init__(self, endpoint, shape, dtype):
        """Create a GRPC Reader.

        Args:
            endpoint: A `tf.string` tensor containing one or more endpoints.
        """
        with tf.name_scope("GRPCStreamIODataset"):
            shape = tf.cast(shape, tf.int64)

            resource = core_ops.io_grpc_readable_init(endpoint)

            self._resource = resource
            self._shape = tf.cast(shape, tf.int64)
            self._dtype = dtype

            step = 1
            indices_start = tf.data.Dataset.range(0, shape[0], step)
            indices_stop = indices_start.skip(1).concatenate(
                tf.data.Dataset.from_tensor_slices([shape[0]])
            )
            dataset = tf.data.Dataset.zip((indices_start, indices_stop))

            def f(start, stop):
                shape = tf.concat(
                    [tf.convert_to_tensor([stop - start], tf.int64), self._shape[1:]],
                    axis=0,
                )
                return core_ops.io_grpc_readable_read(
                    self._resource, start=start, shape=shape, dtype=self._dtype
                )

            dataset = dataset.map(f)
            dataset = dataset.unbatch()

            self._dataset = dataset
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    @staticmethod
    def from_numpy(a, internal=False):
        """from_numpy"""
        assert internal

        from tensorflow_io.core.python.experimental import (  # pylint: disable=import-outside-toplevel
            grpc_endpoint,
        )

        grpc_server = grpc_endpoint.GRPCEndpoint(a)
        grpc_server.start()
        endpoint = grpc_server.endpoint()
        print("ENDPOINT: ", endpoint)
        dtype = a.dtype
        shape = list(a.shape)
        dataset = GRPCStreamIODataset(endpoint, shape, dtype)
        dataset._grpc_server = grpc_server  # pylint: disable=protected-access
        return dataset

    def __del__(self):
        if hasattr(self, "_grpc_server") and self._grpc_server is not None:
            self._grpc_server.stop()

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
