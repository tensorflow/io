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
"""FFmpegDataset"""

import uuid
import sys

import tensorflow as tf
from tensorflow_io.core.python.ops import io_dataset_ops


class FFmpegAudioGraphIODataset(tf.data.Dataset):
    """FFmpegAudioGraphIODataset"""

    def __init__(self, resource, dtype, internal=True):
        """FFmpegAudioGraphIODataset."""
        with tf.name_scope("FFmpegAudioGraphIODataset"):
            from tensorflow_io.core.python.ops import (  # pylint: disable=import-outside-toplevel
                ffmpeg_ops,
            )

            assert internal

            self._resource = resource
            dataset = tf.data.Dataset.range(0, sys.maxsize)
            dataset = dataset.map(lambda e: e == 0)
            dataset = dataset.map(
                lambda reset: ffmpeg_ops.io_ffmpeg_audio_readable_next(
                    resource, reset, dtype=dtype
                )
            )
            dataset = dataset.apply(
                tf.data.experimental.take_while(lambda v: tf.greater(tf.shape(v)[0], 0))
            )
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


class FFmpegVideoGraphIODataset(tf.data.Dataset):
    """FFmpegVideoGraphIODataset"""

    def __init__(self, resource, dtype, internal=True):
        """FFmpegVideoGraphIODataset."""
        with tf.name_scope("FFmpegVideoGraphIODataset"):
            from tensorflow_io.core.python.ops import (  # pylint: disable=import-outside-toplevel
                ffmpeg_ops,
            )

            assert internal

            self._resource = resource
            dataset = tf.data.Dataset.range(0, sys.maxsize)
            dataset = dataset.map(lambda e: e == 0)
            dataset = dataset.map(
                lambda reset: ffmpeg_ops.io_ffmpeg_video_readable_next(
                    resource, reset, dtype=dtype
                )
            )
            dataset = dataset.apply(
                tf.data.experimental.take_while(lambda v: tf.greater(tf.shape(v)[0], 0))
            )
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


class _FFmpegIODatasetFunction:
    def __init__(self, function, resource, component, shape, dtype):
        self._function = function
        self._resource = resource
        self._component = component
        self._shape = shape
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


class FFmpegIODataset(io_dataset_ops._IODataset):  # pylint: disable=protected-access
    """FFmpegIODataset"""

    def __init__(self, filename, stream, internal=True):
        """FFmpegIODataset."""
        with tf.name_scope("FFmpegIODataset") as scope:
            from tensorflow_io.core.python.ops import (  # pylint: disable=import-outside-toplevel
                ffmpeg_ops,
            )

            resource, _ = ffmpeg_ops.io_ffmpeg_readable_init(
                filename,
                container=scope,
                shared_name="{}/{}".format(filename, uuid.uuid4().hex),
            )
            shape, dtype, _ = ffmpeg_ops.io_ffmpeg_readable_spec(resource, stream)
            shape = tf.TensorShape([None if e < 0 else e for e in shape.numpy()])
            dtype = tf.as_dtype(dtype.numpy())
            capacity = 1 if stream.startswith("v:") else 4096
            super().__init__(
                _FFmpegIODatasetFunction(
                    ffmpeg_ops.io_ffmpeg_readable_read, resource, stream, shape, dtype
                ),
                capacity=capacity,
                internal=internal,
            )
