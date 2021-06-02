# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Deprecated Kafka Dataset.

@@KafkaOutputSequence
@@KafkaDataset
@@write_kafka
"""


from tensorflow.python.util.all_util import remove_undocumented

from tensorflow_io.core.python.ops.kafka_deprecated_dataset_ops import (
    KafkaOutputSequence,
)
from tensorflow_io.core.python.ops.kafka_deprecated_dataset_ops import KafkaDataset
from tensorflow_io.core.python.ops.kafka_deprecated_dataset_ops import write_kafka

_allowed_symbols = [
    "KafkaOutputSequence",
    "KafkaDataset",
    "write_kafka",
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
