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
"""PcapDataset"""

import sys
import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class PcapIODataset(tf.data.Dataset):
    """PcapIODataset"""

    def __init__(self, filename, internal=True, **kwargs):
        if not internal:
            raise ValueError(
                "PcapIODataset constructor is private; please use one "
                "of the factory methods instead (e.g., "
                "IODataset.from_pcap())"
            )
        with tf.name_scope("PcapIODataset") as scope:
            capacity = kwargs.get("capacity", 4096)
            resource = core_ops.io_pcap_readable_init(
                filename,
                container=scope,
                shared_name="{}/{}".format(filename, uuid.uuid4().hex),
            )

            dataset = tf.data.Dataset.range(0, sys.maxsize, capacity)
            dataset = dataset.map(
                lambda index: core_ops.io_pcap_readable_read(
                    resource, start=index, stop=index + capacity
                )
            )
            dataset = dataset.apply(
                tf.data.experimental.take_while(
                    lambda v: tf.greater(tf.shape(v.value)[0], 0)
                )
            )
            dataset = dataset.map(lambda v: (v.label, v.value))
            dataset = dataset.unbatch()

            self._capacity = capacity
            self._resource = resource
            self._dataset = dataset
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
