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
"""PulsarDataset"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class PulsarIODataset(tf.data.Dataset):
    """PulsarIODataset"""

    def __init__(
        self,
        service_url,
        topic,
        subscription,
        timeout,
        ack_grouping_time=-1,
        poll_timeout=100,
    ):
        """Creates a `PulsarIODataset` from pulsar server with a subscription

        Args:
          service_url: A `tf.string` tensor containing the service url of pulsar broker.
            For example: "pulsar://localhost:6650".
          topic: A `tf.string` tensor containing the topic name.
          subscription: A `tf.string` tensor containing the subscription name.
          ack_grouping_time: A `tf.int64` tensor containing the ack grouping time.
            If it's non-negative, each time a message was received, the consumer would add it to
            a batch and acknowledge all pending messages later. `ack_grouping_time` is the interval
            between two acknowledgements in milliseconds.
            If it's negative, the message would be acknowledged immediately.
            Default: -1
          timeout: A `tf.int64` tensor containing the timeout in milliseconds. If no messages
            were received after `timeout` milliseconds, we can treat it as no more messages.
            `timeout` must be positive.
          poll_timeout: A `tf.int64` tensor containing the poll timeout in milliseconds. The
            pulsar consumer would try to receive a single message with the poll timeout, if no
            message was received, it would try again until `timeout` exceeds.
            `poll_timeout` must be positive and not larger than `timeout`.
            Default: 100
        """
        with tf.name_scope("PulsarIODataset"):
            if timeout <= 0:
                raise ValueError(
                    "Invalid timeout value: {}, must be > 0".format(timeout)
                )

            if poll_timeout <= 0:
                raise ValueError(
                    "Invalid poll_timeout value: {}, must be > 0".format(poll_timeout)
                )

            if poll_timeout > timeout:
                raise ValueError(
                    "Invalid poll_timeout value: {}, must be <= timeout({})".format(
                        poll_timeout, timeout
                    )
                )

            resource = core_ops.io_pulsar_readable_init(
                service_url, topic, subscription, ack_grouping_time
            )
            self._resource = resource
            dataset = tf.data.experimental.Counter()
            dataset = dataset.map(
                lambda i: core_ops.io_pulsar_readable_next(
                    input=self._resource, timeout=timeout, poll_timeout=poll_timeout
                )
            )
            dataset = dataset.apply(
                tf.data.experimental.take_while(
                    lambda v: tf.greater(v.continue_fetch, 0)
                )
            )
            dataset = dataset.map(lambda v: (v.message, v.key))
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
