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
"""TextIOLayer"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class TextIOLayer(tf.keras.layers.Layer):
    """TextIOLayer"""

    # =============================================================================
    # TextIOLayer
    # =============================================================================
    def __init__(self, filename):
        """Obtain a text file IO layer to be used with tf.keras."""
        self._resource = core_ops.io_file_init(filename)
        super().__init__(trainable=False)

    def sync(self):
        core_ops.io_file_sync(self._resource)

    def call(self, inputs):  # pylint: disable=arguments-differ
        content = tf.reshape(inputs, [tf.shape(inputs)[0], -1])
        if inputs.dtype != tf.string:
            content = tf.strings.as_string(content)
        content = tf.strings.reduce_join(content, axis=1, separator=",")
        content = content + tf.constant(["\n"])
        return core_ops.io_file_call(content, False, self._resource)
