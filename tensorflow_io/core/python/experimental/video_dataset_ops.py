# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""VideoCaptureDataset"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class VideoCaptureIODataset(tf.data.Dataset):
    """VideoCaptureIODataset"""

    def __init__(self, device, internal=True):
        """VideoCaptureIODataset"""
        with tf.name_scope("VideoCaptureIODataset"):
            assert internal

            resource = core_ops.io_video_capture_readable_init(device)

            self._resource = resource

            dataset = tf.data.experimental.Counter()
            dataset = dataset.map(
                lambda i: core_ops.io_video_capture_readable_read(self._resource, i)
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


class VideoIODataset(tf.data.Dataset):
    """VideoIODataset"""

    def __init__(self, filename, internal=True):
        """VideoIODataset"""
        with tf.name_scope("VideoIODataset"):
            assert internal

            resource = core_ops.io_video_readable_init(filename)

            self._resource = resource

            dataset = tf.data.experimental.Counter()
            dataset = dataset.map(
                lambda i: core_ops.io_video_readable_read(self._resource, i)
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
