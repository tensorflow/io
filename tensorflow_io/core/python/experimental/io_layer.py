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
"""IOLayer"""

import tensorflow as tf
from tensorflow_io.core.python.experimental import text_io_layer_ops
from tensorflow_io.core.python.experimental import kafka_io_layer_ops


class IOLayer(tf.keras.layers.Layer):
    """IOLayer"""

    # =============================================================================
    # IOLayer (identity)
    # =============================================================================
    def __init__(self):
        """Obtain an identity layer to be used with tf.keras."""
        super().__init__(trainable=False)

    def call(self, inputs):  # pylint: disable=arguments-differ
        return tf.identity(inputs)

    # =============================================================================
    # TextIOLayer
    # =============================================================================

    @classmethod
    def text(cls, filename):
        """Obtain a TextIOLayer to be used with tf.keras.

        Args:
          filename: A `string` Tensor of the filename.

        Returns:
          A class of `TextIOLayer`.
        """
        return text_io_layer_ops.TextIOLayer(filename)

    # =============================================================================
    # KafkaIOLayer
    # =============================================================================

    @classmethod
    def kafka(cls, topic, partition=0, servers=None, configuration=None):
        """Obtain a KafkaIOLayer to be used with tf.keras.

        Args:
          topic: A `tf.string` tensor containing topic.
          partition: A `tf.int32` tensor containing partition.
          servers: A list of bootstrap servers.
          configurations: A `tf.string` tensor containing global configuration
            properties in [Key=Value] format,eg.
            ["enable.auto.commit=false", "heartbeat.interval.ms=2000"],
            please refer to 'Global configuration properties'
            in librdkafka doc.

        Returns:
          A class of `KafkaIOLayer`.
        """
        return kafka_io_layer_ops.KafkaIOLayer(topic, partition, servers, configuration)
