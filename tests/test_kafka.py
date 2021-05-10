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


import sys
import time
import pytest
import numpy as np
import threading

import tensorflow as tf
import tensorflow_io as tfio


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
    import tensorflow_io.kafka as kafka_io

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
    import tensorflow_io.kafka as kafka_io

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
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=earliest",
        ],
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
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=earliest",
        ],
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
    import tensorflow_io.kafka as kafka_io

    # Write new messages to the topic
    for i in range(10, 100):
        message = "D{}".format(i)
        kafka_io.write_kafka(message=message, topic="key-partition-test")
    # Read only the newly sent 90 messages
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
    import tensorflow_io.kafka as kafka_io

    # Write new messages to the topic
    for i in range(10, 100):
        message = "D{}".format(i)
        kafka_io.write_kafka(message=message, topic="key-test")
    # Read only the newly sent 90 messages
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
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=earliest",
        ],
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
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=earliest",
        ],
    )
    assert np.all(
        sorted([k.numpy() for (k, _) in dataset])
        == sorted([("D" + str(i)).encode() for i in range(100)] * 2)
    )


def test_kafka_group_io_dataset_auto_offset_reset():
    """Test the functionality of the `auto.offset.reset` configuration
    at global and topic level"""

    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["key-partition-test"],
        group_id="cgglobaloffsetearliest",
        servers="localhost:9092",
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=earliest",
        ],
    )
    assert np.all(
        sorted([k.numpy() for (k, _) in dataset])
        == sorted([("D" + str(i)).encode() for i in range(100)])
    )

    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["key-partition-test"],
        group_id="cgglobaloffsetlatest",
        servers="localhost:9092",
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=latest",
        ],
    )
    assert np.all(sorted([k.numpy() for (k, _) in dataset]) == [])

    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["key-partition-test"],
        group_id="cgtopicoffsetearliest",
        servers="localhost:9092",
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "conf.topic.auto.offset.reset=earliest",
        ],
    )
    assert np.all(
        sorted([k.numpy() for (k, _) in dataset])
        == sorted([("D" + str(i)).encode() for i in range(100)])
    )

    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["key-partition-test"],
        group_id="cgtopicoffsetlatest",
        servers="localhost:9092",
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "conf.topic.auto.offset.reset=latest",
        ],
    )
    assert np.all(sorted([k.numpy() for (k, _) in dataset]) == [])


def test_kafka_group_io_dataset_invalid_stream_timeout():
    """Test the functionality of the KafkaGroupIODataset when the
    consumer is configured to have an invalid stream_timeout value which is
    less than the message_timeout value.
    NOTE: The default value for message_timeout=5000
    """

    STREAM_TIMEOUT = -20
    try:
        tfio.experimental.streaming.KafkaGroupIODataset(
            topics=["key-partition-test", "key-test"],
            group_id="cgteststreaminvalid",
            servers="localhost:9092",
            stream_timeout=STREAM_TIMEOUT,
            configuration=["session.timeout.ms=7000", "max.poll.interval.ms=8000"],
        )
    except ValueError as e:
        assert str(
            e
        ) == "Invalid stream_timeout value: {} ,set it to -1 to block indefinitely.".format(
            STREAM_TIMEOUT
        )


def test_kafka_group_io_dataset_stream_timeout_check():
    """Test the functionality of the KafkaGroupIODataset when the
    consumer is configured to have a valid stream_timeout value and thus waits
    for the new messages from kafka.
    NOTE: The default value for message_timeout=5000
    """
    import tensorflow_io.kafka as kafka_io

    def write_messages_background():
        # Write new messages to the topic in a background thread
        time.sleep(6)
        for i in range(100, 200):
            message = "D{}".format(i)
            kafka_io.write_kafka(message=message, topic="key-partition-test")

    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["key-partition-test"],
        group_id="cgteststreamvalid",
        servers="localhost:9092",
        stream_timeout=20000,
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=earliest",
        ],
    )

    # start writing the new messages to kafka using the background job.
    # the job sleeps for some time (< stream_timeout) and then writes the
    # messages into the topic.
    thread = threading.Thread(target=write_messages_background, args=())
    thread.daemon = True
    thread.start()

    # At the end, after the timeout has occurred, we must have the old 100 messages
    # along with the new 100 messages
    assert np.all(
        sorted([k.numpy() for (k, _) in dataset])
        == sorted([("D" + str(i)).encode() for i in range(200)])
    )


def test_kafka_batch_io_dataset():
    """Test the functionality of the KafkaBatchIODataset by training a model
    directly on the incoming kafka message batch(of type tf.data.Dataset), in an
    online-training fashion.

    NOTE: This kind of dataset is suitable in scenarios where the 'keys' of 'messages'
        act as labels. If not, additional transformations are required.
    """

    dataset = tfio.experimental.streaming.KafkaBatchIODataset(
        topics=["mini-batch-test"],
        group_id="cgminibatch",
        servers=None,
        stream_timeout=5000,
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=earliest",
        ],
    )

    NUM_COLUMNS = 1
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(NUM_COLUMNS,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    assert issubclass(type(dataset), tf.data.Dataset)
    for mini_d in dataset:
        mini_d = mini_d.map(
            lambda m, k: (
                tf.strings.to_number(m, out_type=tf.float32),
                tf.strings.to_number(k, out_type=tf.float32),
            )
        ).batch(2)
        assert issubclass(type(mini_d), tf.data.Dataset)
        # Fits the model as long as the data keeps on streaming
        model.fit(mini_d, epochs=5)
