# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Kafka Output Sequence."""


import time
import pytest
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow_io.kafka.python.ops import (
    kafka_ops,
)  # pylint: disable=wrong-import-position
import tensorflow_io.kafka as kafka_io  # pylint: disable=wrong-import-position


def test_kafka_io_tensor():
    kafka = tfio.IOTensor.from_kafka("test")
    assert kafka.dtype == tf.string
    assert kafka.shape.as_list() == [None]
    assert np.all(
        kafka.to_tensor().numpy() == [("D" + str(i)).encode() for i in range(10)]
    )
    assert len(kafka.to_tensor()) == 10


@pytest.mark.skip(reason="TODO")
def test_kafka_output_sequence():
    """Test case based on fashion mnist tutorial"""
    fashion_mnist = tf.keras.datasets.fashion_mnist
    ((train_images, train_labels), (test_images, _)) = fashion_mnist.load_data()

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(train_images, train_labels, epochs=5)

    class OutputCallback(tf.keras.callbacks.Callback):
        """KafkaOutputCallback"""

        def __init__(
            self, batch_size, topic, servers
        ):  # pylint: disable=super-init-not-called
            self._sequence = kafka_ops.KafkaOutputSequence(topic=topic, servers=servers)
            self._batch_size = batch_size

        def on_predict_batch_end(self, batch, logs=None):
            index = batch * self._batch_size
            for outputs in logs["outputs"]:
                for output in outputs:
                    self._sequence.setitem(index, class_names[np.argmax(output)])
                    index += 1

        def flush(self):
            self._sequence.flush()

    channel = "e{}e".format(time.time())
    topic = "test_" + channel

    # By default batch size is 32
    output = OutputCallback(32, topic, "localhost")
    predictions = model.predict(test_images, callbacks=[output])
    output.flush()

    predictions = [class_names[v] for v in np.argmax(predictions, axis=1)]

    # Reading from `test_e(time)e` we should get the same result
    dataset = tfio.kafka.KafkaDataset(topics=[topic], group="test", eof=True)
    for entry, prediction in zip(dataset, predictions):
        assert entry.numpy() == prediction.encode()


def test_avro_kafka_dataset():
    """test_avro_kafka_dataset"""
    schema = (
        '{"type":"record","name":"myrecord","fields":['
        '{"name":"f1","type":"string"},'
        '{"name":"f2","type":"long"},'
        '{"name":"f3","type":["null","string"],"default":null}'
        "]}"
    )
    dataset = kafka_io.KafkaDataset(["avro-test:0"], group="avro-test", eof=True)
    # remove kafka framing
    dataset = dataset.map(lambda e: tf.strings.substr(e, 5, -1))
    # deserialize avro
    dataset = dataset.map(
        lambda e: tfio.experimental.serialization.decode_avro(e, schema=schema)
    )
    entries = [(e["f1"], e["f2"], e["f3"]) for e in dataset]
    np.all(entries == [("value1", 1, ""), ("value2", 2, ""), ("value3", 3, "")])


def test_avro_kafka_dataset_with_resource():
    """test_avro_kafka_dataset_with_resource"""
    schema = (
        '{"type":"record","name":"myrecord","fields":['
        '{"name":"f1","type":"string"},'
        '{"name":"f2","type":"long"},'
        '{"name":"f3","type":["null","string"],"default":null}'
        ']}"'
    )
    schema_resource = kafka_io.decode_avro_init(schema)
    dataset = kafka_io.KafkaDataset(["avro-test:0"], group="avro-test", eof=True)
    # remove kafka framing
    dataset = dataset.map(lambda e: tf.strings.substr(e, 5, -1))
    # deserialize avro
    dataset = dataset.map(
        lambda e: kafka_io.decode_avro(
            e, schema=schema_resource, dtype=[tf.string, tf.int64, tf.string]
        )
    )
    entries = [(f1.numpy(), f2.numpy(), f3.numpy()) for (f1, f2, f3) in dataset]
    np.all(entries == [("value1", 1), ("value2", 2), ("value3", 3)])


def test_kafka_stream_dataset():
    dataset = tfio.IODataset.stream().from_kafka("test").batch(2)
    assert np.all(
        [k.numpy().tolist() for (k, _) in dataset]
        == np.asarray([("D" + str(i)).encode() for i in range(10)]).reshape((5, 2))
    )


def test_kafka_io_dataset():
    dataset = tfio.IODataset.from_kafka(
        "test", configuration=["fetch.min.bytes=2"]
    ).batch(2)
    # repeat multiple times will result in the same result
    for _ in range(5):
        assert np.all(
            [k.numpy().tolist() for (k, _) in dataset]
            == np.asarray([("D" + str(i)).encode() for i in range(10)]).reshape((5, 2))
        )


def test_avro_encode_decode():
    """test_avro_encode_decode"""
    schema = (
        '{"type":"record","name":"myrecord","fields":'
        '[{"name":"f1","type":"string"},{"name":"f2","type":"long"}]}'
    )
    value = [("value1", 1), ("value2", 2), ("value3", 3)]
    f1 = tf.cast([v[0] for v in value], tf.string)
    f2 = tf.cast([v[1] for v in value], tf.int64)
    message = tfio.experimental.serialization.encode_avro([f1, f2], schema=schema)
    entries = tfio.experimental.serialization.decode_avro(message, schema=schema)
    assert np.all(entries["f1"].numpy() == f1.numpy())
    assert np.all(entries["f2"].numpy() == f2.numpy())


def test_kafka_group_io_dataset_primary_cg():
    """Test the functionality of the KafkaGroupIODataset when the consumer group
    is being newly created.

    NOTE: After the kafka cluster is setup during the testing phase, 10 messages
    are written to the 'key-partition-test' topic with 5 in each partition
    (topic created with 2 partitions, the messages are split based on the keys).
    And the same 10 messages are written into the 'key-test' topic (topic created
    with 1 partition, so no splitting of the messages based on the keys).

    K0:D0, K1:D1, K0:D2, K1:D3, K0:D4, K1:D5, K0:D6, K1:D7, K0:D8, K1:D9.

    Here, messages D0, D2, D4, D6 and D8 are written into partition 0 and the rest are written
    into partition 1.

    Also, since the messages are read from different partitions, the order of retrieval may not be
    the same as storage. Thus, we sort and compare.
    """
    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["key-partition-test"],
        group_id="cgtestprimary",
        servers="localhost:9092",
        configuration=["session.timeout.ms=7000", "max.poll.interval.ms=8000"],
    )
    assert np.all(
        sorted([k.numpy() for (k, _) in dataset])
        == sorted([("D" + str(i)).encode() for i in range(10)])
    )


def test_kafka_group_io_dataset_primary_cg_no_lag():
    """Test the functionality of the KafkaGroupIODataset when the
    consumer group has read all the messages and committed the offsets.
    """
    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["key-partition-test"],
        group_id="cgtestprimary",
        servers="localhost:9092",
        configuration=["session.timeout.ms=7000", "max.poll.interval.ms=8000"],
    )
    assert np.all(sorted([k.numpy() for (k, _) in dataset]) == [])


def test_kafka_group_io_dataset_primary_cg_new_topic():
    """Test the functionality of the KafkaGroupIODataset when the existing
    consumer group reads data from a new topic.
    """
    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["key-test"],
        group_id="cgtestprimary",
        servers="localhost:9092",
        configuration=["session.timeout.ms=7000", "max.poll.interval.ms=8000"],
    )
    assert np.all(
        sorted([k.numpy() for (k, _) in dataset])
        == sorted([("D" + str(i)).encode() for i in range(10)])
    )


def test_kafka_group_io_dataset_resume_primary_cg():
    """Test the functionality of the KafkaGroupIODataset when the
    consumer group is yet to catch up with the newly added messages only
    (Instead of reading from the beginning).
    """

    # Write new messages to the topic
    for i in range(10, 100):
        message = "D{}".format(i)
        kafka_io.write_kafka(message=message, topic="key-partition-test")

    # Read only the newly sent 100 messages
    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["key-partition-test"],
        group_id="cgtestprimary",
        servers="localhost:9092",
        configuration=["session.timeout.ms=7000", "max.poll.interval.ms=8000"],
    )
    assert np.all(
        sorted([k.numpy() for (k, _) in dataset])
        == sorted([("D" + str(i)).encode() for i in range(10, 100)])
    )


def test_kafka_group_io_dataset_resume_primary_cg_new_topic():
    """Test the functionality of the KafkaGroupIODataset when the
    consumer group is yet to catch up with the newly added messages only
    (Instead of reading from the beginning) from the new topic.
    """

    # Write new messages to the topic
    for i in range(10, 100):
        message = "D{}".format(i)
        kafka_io.write_kafka(message=message, topic="key-test")

    # Read only the newly sent 100 messages
    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["key-test"],
        group_id="cgtestprimary",
        servers="localhost:9092",
        configuration=["session.timeout.ms=7000", "max.poll.interval.ms=8000"],
    )
    assert np.all(
        sorted([k.numpy() for (k, _) in dataset])
        == sorted([("D" + str(i)).encode() for i in range(10, 100)])
    )


def test_kafka_group_io_dataset_secondary_cg():
    """Test the functionality of the KafkaGroupIODataset when a
    secondary consumer group is created and is yet to catch up all the messages,
    from the beginning.
    """

    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["key-partition-test"],
        group_id="cgtestsecondary",
        servers="localhost:9092",
        configuration=["session.timeout.ms=7000", "max.poll.interval.ms=8000"],
    )
    assert np.all(
        sorted([k.numpy() for (k, _) in dataset])
        == sorted([("D" + str(i)).encode() for i in range(100)])
    )


def test_kafka_group_io_dataset_tertiary_cg_multiple_topics():
    """Test the functionality of the KafkaGroupIODataset when a new
    consumer group reads data from multiple topics from the beginning.
    """

    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["key-partition-test", "key-test"],
        group_id="cgtesttertiary",
        servers="localhost:9092",
        configuration=["session.timeout.ms=7000", "max.poll.interval.ms=8000"],
    )
    assert np.all(
        sorted([k.numpy() for (k, _) in dataset])
        == sorted([("D" + str(i)).encode() for i in range(100)] * 2)
    )
