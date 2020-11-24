# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for Pulsar Dataset"""

import numpy as np

import tensorflow as tf
import tensorflow_io as tfio


default_pulsar_timeout = 5000


def test_pulsar_simple_messages():
    """Test consuming simple messages from a Pulsar topic with PulsarIODataset. 

    NOTE: After the pulsar standalone is setup during the testing phase, 6 messages
    (D0, D1, ..., D5) are sent to the `test` topic.
    """

    dataset = tfio.experimental.streaming.PulsarIODataset(
        service_url="pulsar://localhost:6650",
        topic="test",
        subscription="subscription-0",
        timeout=default_pulsar_timeout,
    )
    assert np.all(
        [k.numpy() for (k, _) in dataset] == [("D" + str(i)).encode() for i in range(6)]
    )


def test_pulsar_keyed_messages():
    """Test consuming keyed messages from a Pulsar topic with PulsarIODataset

    NOTE: After the pulsar standalone is setup during the testing phase, 6 messages
    are sent to the `test` topic:

    K0:D0, K1:D1, K0:D2, K1:D3, K0:D4, K1:D5.
    """

    dataset = tfio.experimental.streaming.PulsarIODataset(
        service_url="pulsar://localhost:6650",
        topic="key-test",
        subscription="subscription-0",
        timeout=default_pulsar_timeout,
    )
    kv = dict()
    for (msg, key) in dataset:
        kv.setdefault(key.numpy().decode(), []).append(msg.numpy())
    assert kv["K0"] == [("D" + str(i)).encode() for i in range(0, 6, 2)]
    assert kv["K1"] == [("D" + str(i)).encode() for i in range(1, 6, 2)]


def test_pulsar_resubscribe():
    """Test resubscribing the same topic.

    If a topic is resubscribed with an existed subscription, the consumer will continue
    consuming from the last position.

    NOTE: This test must be run after `test_pulsar_simple_messages`.
    """

    topic = "test"
    writer = tfio.experimental.streaming.PulsarWriter(
        service_url="pulsar://localhost:6650", topic=topic
    )
    # 1. Append new messages to topic
    for i in range(6, 10):
        writer.write("D" + str(i))
    writer.flush()

    # 2. Use the same subscription with `test_pulsar_simple_messages` to continue consuming
    dataset = tfio.experimental.streaming.PulsarIODataset(
        service_url="pulsar://localhost:6650",
        topic=topic,
        subscription="subscription-0",
        timeout=default_pulsar_timeout,
    )
    assert np.all(
        [k.numpy() for (k, _) in dataset]
        == [("D" + str(i)).encode() for i in range(6, 10)]
    )

    # 3. Use another subscription to consume messages from beginning
    dataset = tfio.experimental.streaming.PulsarIODataset(
        service_url="pulsar://localhost:6650",
        topic=topic,
        subscription="subscription-1",
        timeout=default_pulsar_timeout,
    )
    assert np.all(
        [k.numpy() for (k, _) in dataset]
        == [("D" + str(i)).encode() for i in range(10)]
    )


def test_pulsar_invalid_arguments():
    """Test the invalid arguments when a PulsarIODataset is created

    The following cases are included:
    1. timeout is non-positive
    2. poll_timeout is non-positive
    3. poll_timeout is larger than timeout
    """

    INVALID_TIMEOUT = -123
    try:
        tfio.experimental.streaming.PulsarIODataset(
            service_url="pulsar://localhost:6650",
            topic="test",
            subscription="subscription-0",
            timeout=INVALID_TIMEOUT,
        )
    except ValueError as e:
        assert str(e) == "Invalid timeout value: {}, must be > 0".format(
            INVALID_TIMEOUT
        )

    VALID_TIMEOUT = default_pulsar_timeout
    INVALID_POLL_TIMEOUT = -45
    try:
        tfio.experimental.streaming.PulsarIODataset(
            service_url="pulsar://localhost:6650",
            topic="test",
            subscription="subscription-0",
            timeout=VALID_TIMEOUT,
            poll_timeout=INVALID_POLL_TIMEOUT,
        )
    except ValueError as e:
        assert str(e) == "Invalid poll_timeout value: {}, must be > 0".format(
            INVALID_POLL_TIMEOUT
        )

    LARGE_POLL_TIMEOUT = VALID_TIMEOUT + 1
    try:
        tfio.experimental.streaming.PulsarIODataset(
            service_url="pulsar://localhost:6650",
            topic="test",
            subscription="subscription-0",
            timeout=VALID_TIMEOUT,
            poll_timeout=LARGE_POLL_TIMEOUT,
        )
    except ValueError as e:
        assert str(
            e
        ) == "Invalid poll_timeout value: {}, must be <= timeout({})".format(
            LARGE_POLL_TIMEOUT, VALID_TIMEOUT
        )


def test_pulsar_write_simple_messages():
    """Test writing simple messages to a Pulsar topic with PulsarWriter
    """

    topic = "test-write-simple-messages"
    writer = tfio.experimental.streaming.PulsarWriter(
        service_url="pulsar://localhost:6650", topic=topic
    )
    # 1. Write 10 messages
    for i in range(10):
        writer.write("msg-" + str(i))
    writer.flush()

    # 2. Consume messages and verify
    dataset = tfio.experimental.streaming.PulsarIODataset(
        service_url="pulsar://localhost:6650",
        topic=topic,
        subscription="subscription-0",
        timeout=default_pulsar_timeout,
    )
    assert np.all(
        [k.numpy() for (k, _) in dataset]
        == [("msg-" + str(i)).encode() for i in range(10)]
    )


def test_pulsar_write_keyed_messages():
    """Test writing keyed messages to a Pulsar topic with PulsarWriter
    """

    topic = "test-write-keyed-messages"
    writer = tfio.experimental.streaming.PulsarWriter(
        service_url="pulsar://localhost:6650", topic=topic
    )
    # 1. Write 10 keyed messages, the key set is 0,1,2,0,1,2,...
    for i in range(10):
        value = "msg-" + str(i)
        key = str(i % 3)
        writer.write(value=value, key=key)
    writer.flush()

    # 2. Consume messages and verify
    dataset = tfio.experimental.streaming.PulsarIODataset(
        service_url="pulsar://localhost:6650",
        topic=topic,
        subscription="subscription-0",
        timeout=default_pulsar_timeout,
    )
    kv = dict()
    for (msg, key) in dataset:
        kv.setdefault(key.numpy().decode(), []).append(msg.numpy())
    assert kv["0"] == [("msg-" + str(i)).encode() for i in range(0, 10, 3)]
    assert kv["1"] == [("msg-" + str(i)).encode() for i in range(1, 10, 3)]
    assert kv["2"] == [("msg-" + str(i)).encode() for i in range(2, 10, 3)]


if __name__ == "__main__":
    test.main()
