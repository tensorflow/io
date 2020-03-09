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
"""KafkaOutputSequence."""

from tensorflow_io.core.python.ops import core_ops


class KafkaOutputSequence():
  """KafkaOutputSequence"""

  def __init__(self, topic, servers="localhost", configuration=None):
    """Create a `KafkaOutputSequence`.
    """
    self._topic = topic
    metadata = list(configuration or [])
    if servers is not None:
      metadata.append("bootstrap.servers=%s" % servers)
    self._resource = core_ops.io_kafka_output_sequence(
        topic=topic, metadata=metadata)

  def setitem(self, index, item):
    core_ops.io_kafka_output_sequence_set_item(self._resource, index, item)

  def flush(self):
    core_ops.io_kafka_output_sequence_flush(self._resource)
