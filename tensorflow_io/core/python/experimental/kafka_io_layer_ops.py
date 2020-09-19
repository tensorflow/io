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
"""KafkaIOLayer"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class KafkaIOLayer(tf.keras.layers.Layer):
    """KafkaIOLayer"""

    # =============================================================================
    # KafkaIOLayer
    # =============================================================================
    def __init__(self, topic, partition, servers, configurations):
        """Obtain a Kafka IO layer to be used with tf.keras."""
        metadata = list(configurations or [])
        if servers is not None:
            metadata.append("bootstrap.servers=%s" % servers)
        self._resource = core_ops.io_layer_kafka_init(topic, partition, metadata)
        super().__init__(trainable=False)

    def sync(self):
        core_ops.io_layer_kafka_sync(self._resource)

    def call(self, inputs):  # pylint: disable=arguments-differ
        content = tf.reshape(inputs, [tf.shape(inputs)[0], -1])
        if inputs.dtype != tf.string:
            content = tf.strings.as_string(content)
        content = tf.strings.reduce_join(content, axis=1, separator=",")
        return core_ops.io_layer_kafka_call(inputs, content, self._resource)
