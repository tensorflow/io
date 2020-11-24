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
"""PulsarWriter"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class PulsarWriter:
    """PulsarWriter"""

    def __init__(self, service_url, topic):
        """Creates a `PulsarWriter` for writing messages to a pulsar topic

        Args:
          service_url: A `tf.string` tensor containing the service url of pulsar broker.
            For example: "pulsar://localhost:6650".
          topic: A `tf.string` tensor containing the topic name.
        """
        with tf.name_scope("PulsarWriter"):
            resource = core_ops.io_pulsar_writable_init(service_url, topic)
            self._resource = resource

    def write(self, value, key=""):
        """Write a message to pulsar topic asynchronously

        Args:
          value: A `tf.string` tensor containing the value of message
          key: A `tf.string` tensor containing the key of message, if it's an empty string, the message will have no key.
            Default: ""
        """
        return core_ops.io_pulsar_writable_write(self._resource, value, key)

    def flush(self):
        """Flush the queued messages, it will wait async write operations completed."""
        return core_ops.io_pulsar_writable_flush(self._resource)
