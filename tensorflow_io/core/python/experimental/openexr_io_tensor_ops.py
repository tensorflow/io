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
"""EXRIOTensor"""

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops


class EXRPartIOTensor(io_tensor_ops._TableIOTensor):  # pylint: disable=protected-access
    """EXRPartIOTensor"""

    # =============================================================================
    # Constructor (private)
    # =============================================================================
    def __init__(self, spec, columns, values, internal=False):
        with tf.name_scope("EXRPartIOTensor"):
            super().__init__(spec, columns, values, internal=internal)


class EXRIOTensor(
    io_tensor_ops._CollectionIOTensor
):  # pylint: disable=protected-access
    """EXRIOTensor"""

    # =============================================================================
    # Constructor (private)
    # =============================================================================
    def __init__(self, filename, internal=False):
        with tf.name_scope("EXRIOTensor"):
            data = tf.io.read_file(filename)
            shapes, dtypes, channels = core_ops.io_decode_exr_info(data)
            parts = []
            index = 0
            for (shape, dtypes, channels) in zip(
                shapes.numpy(), dtypes.numpy(), channels.numpy()
            ):
                # Remove trailing 0 from dtypes
                while dtypes[-1] == 0:
                    dtypes.pop()
                    channels.pop()
                spec = tuple(
                    [tf.TensorSpec(tf.TensorShape(shape), dtype) for dtype in dtypes]
                )
                columns = [channel.decode() for channel in channels]
                elements = [
                    io_tensor_ops.TensorIOTensor(
                        core_ops.io_decode_exr(data, index, channel, dtype=dtype),
                        internal=internal,
                    )
                    for (channel, dtype) in zip(columns, dtypes)
                ]
                parts.append(
                    EXRPartIOTensor(spec, columns, elements, internal=internal)
                )
                index += 1
            spec = tuple([part.spec for part in parts])
            columns = [i for i, _ in enumerate(parts)]
            super().__init__(spec, columns, parts, internal=internal)
