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
"""KafkaGroupIODatasets"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class KafkaGroupIODataset(tf.data.Dataset):
    """KafkaGroupIODataset"""

    def __init__(
        self,
        topics,
        group_id,
        servers,
        message_timeout=5000,
        stream_timeout=5000,
        configuration=None,
        internal=True,
    ):
        """Creates an `IODataset` from kafka server by joining a consumer group
        and maintaining offsets of all the partitions without explicit initialization.
        If the consumer joins an existing consumer group, it will start fetching
        messages based on the already committed offsets. To start fetching the messages
        from the beginning, please join a different consumer group. The dataset will be prepared
        from the committed/start offset until the last offset.

        NOTE: Cases may arise where the consumer read time out issues arise due to
        the consumer group being in a rebalancing state. In order to address that, please
        set `session.timeout.ms` and `max.poll.interval.ms` values in the configuration tensor
        and try again after the group rebalances. For example: considering your kafka cluster
        has been setup with the default settings, `max.poll.interval.ms` would be `300000ms`.
        It can be changed to `8000ms` to reduce the time between pools. Also, the `session.timeout.ms`
        can be changed to `7000ms`. However, the value for `session.timeout.ms` should be
        according to the following relation:

        - `group.max.session.timeout.ms` in server.properties > `session.timeout.ms` in the
        consumer.properties.
        - `group.min.session.timeout.ms` in server.properties < `session.timeout.ms` in the
        consumer.properties

        Args:
          topics: A `tf.string` tensor containing topic names in [topic] format.
            For example: ["topic1"]
          group_id: The id of the consumer group. For example: cgstream
          servers: An optional list of bootstrap servers.
            For example: `localhost:9092`.
          message_timeout: An optional timeout value (in milliseconds) for retrieving messages
            from kafka. Default value is 5000.
          stream_timeout: An optional timeout value (in milliseconds) to wait for the new messages
            from kafka to be retrieved by the consumers. Default value is 5000.
            NOTE: The `stream_timeout` value should always be greater than or equal to the `message_timeout`.
            value.
          configuration: An optional `tf.string` tensor containing
            configurations in [Key=Value] format.
            Global configuration: please refer to 'Global configuration properties'
              in librdkafka doc. Examples include
              ["enable.auto.commit=false", "heartbeat.interval.ms=2000"]
            Topic configuration: please refer to 'Topic configuration properties'
              in librdkafka doc. Note all topic configurations should be
              prefixed with `configuration.topic.`. Examples include
              ["conf.topic.auto.offset.reset=earliest"]
          internal: Whether the dataset is being created from within the named scope.
            Default: True
        """
        with tf.name_scope("KafkaGroupIODataset"):
            assert internal

            if stream_timeout < message_timeout:
                raise ValueError(
                    "stream_timeout {} is less than the message_timeout {}".format(
                        stream_timeout, message_timeout
                    )
                )
            metadata = list(configuration or [])
            if group_id is not None:
                metadata.append("group.id=%s" % group_id)
            if servers is not None:
                metadata.append("bootstrap.servers=%s" % servers)
            resource = core_ops.io_kafka_group_readable_init(
                topics=topics, metadata=metadata
            )

            self._resource = resource
            dataset = tf.data.experimental.Counter()
            dataset = dataset.map(
                lambda i: core_ops.io_kafka_group_readable_next(
                    input=self._resource,
                    index=i,
                    message_timeout=message_timeout,
                    stream_timeout=stream_timeout,
                )
            )
            dataset = dataset.apply(
                tf.data.experimental.take_while(
                    lambda v: tf.greater(tf.shape(v.message)[0], 0)
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
