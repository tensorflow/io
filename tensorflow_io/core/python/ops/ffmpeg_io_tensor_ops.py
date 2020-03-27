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
"""FFmpegIOTensor"""

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops


class _FFmpegIOTensorFunction:
    """_FFmpegIOTensorFuntion"""

    def __init__(self, function, resource, component, shape, dtype, capacity):
        self._function = function
        self._resource = resource
        self._component = component
        self._shape = shape
        self._dtype = dtype
        self._capacity = capacity
        self._index = 0

    def __call__(self):
        items = self._function(
            self._resource,
            start=self._index,
            stop=self._index + self._capacity,
            component=self._component,
            shape=self._shape,
            dtype=self._dtype,
        )
        self._index += items.shape[0]
        return items


class FFmpegVideoIOTensor(io_tensor_ops.BaseIOTensor):
    """FFmpegVideoIOTensor"""

    def __init__(self, spec, function, internal=False):
        with tf.name_scope("FFmpegVideoIOTensor"):
            super().__init__(spec, function, internal=internal)


class FFmpegAudioIOTensor(io_tensor_ops.BaseIOTensor):
    """FFmpegAudioIOTensor"""

    def __init__(self, spec, function, rate, internal=False):
        with tf.name_scope("FFmpegAudioIOTensor"):
            self._rate = rate
            super().__init__(spec, function, internal=internal)

    @io_tensor_ops._IOTensorMeta  # pylint: disable=protected-access
    def rate(self):
        """The sample `rate` of the audio stream"""
        return self._rate


class FFmpegSubtitleIOTensor(io_tensor_ops.BaseIOTensor):
    """FFmpegSubtitleIOTensor"""

    def __init__(self, spec, function, internal=False):
        with tf.name_scope("FFmpegSubtitleIOTensor"):
            super().__init__(spec, function, internal=internal)


class FFmpegIOTensor(
    io_tensor_ops._CollectionIOTensor
):  # pylint: disable=protected-access
    """FFmpegIOTensor"""

    # =============================================================================
    # Constructor (private)
    # =============================================================================
    def __init__(self, filename, internal=False):
        with tf.name_scope("FFmpegIOTensor") as scope:
            from tensorflow_io.core.python.ops import (  # pylint: disable=import-outside-toplevel
                ffmpeg_ops,
            )

            resource, columns = ffmpeg_ops.io_ffmpeg_readable_init(
                filename,
                container=scope,
                shared_name="{}/{}".format(filename, uuid.uuid4().hex),
            )
            columns = [column.decode() for column in columns.numpy().tolist()]
            elements = []
            for column in columns:
                shape, dtype, rate = ffmpeg_ops.io_ffmpeg_readable_spec(
                    resource, column
                )
                shape = tf.TensorShape([None if e < 0 else e for e in shape.numpy()])
                dtype = tf.as_dtype(dtype.numpy())
                spec = tf.TensorSpec(shape, dtype, column)
                capacity = 1 if column.startswith("v:") else 4096
                function = _FFmpegIOTensorFunction(
                    ffmpeg_ops.io_ffmpeg_readable_read,
                    resource,
                    column,
                    shape,
                    dtype,
                    capacity=capacity,
                )
                function = io_tensor_ops._IOTensorIterablePartitionedFunction(  # pylint: disable=protected-access
                    function, shape
                )
                if column.startswith("v:"):
                    elements.append(
                        FFmpegVideoIOTensor(spec, function, internal=internal)
                    )
                elif column.startswith("a:"):
                    rate = rate.numpy()
                    elements.append(
                        FFmpegAudioIOTensor(spec, function, rate, internal=internal)
                    )
                else:
                    elements.append(
                        FFmpegSubtitleIOTensor(spec, function, internal=internal)
                    )
            spec = tuple([e.spec for e in elements])
            super().__init__(spec, columns, elements, internal=internal)
