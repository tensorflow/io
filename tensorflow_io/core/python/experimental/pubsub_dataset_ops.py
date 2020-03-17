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
"""PubSubDataset"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class PubSubStreamIODataset(tf.data.Dataset):
    """PubSubStreamGraphIODataset"""

    def __init__(self, subscription, endpoint=None, timeout=10000, internal=True):
        """PubSubStreamIODataset."""
        with tf.name_scope("PubSubStreamIODataset"):
            assert internal

            metadata = []
            if endpoint is not None:
                metadata.append("endpoint=%s" % endpoint)
            metadata.append("timeout=%d" % timeout)
            resource = core_ops.io_pub_sub_readable_init(subscription, metadata)

            self._resource = resource

            dataset = tf.data.experimental.Counter()
            dataset = dataset.map(
                lambda i: core_ops.io_pub_sub_readable_read(self._resource, i)
            )
            dataset = dataset.apply(
                tf.data.experimental.take_while(
                    lambda v: tf.greater(tf.shape(v.id)[0], 0)
                )
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
