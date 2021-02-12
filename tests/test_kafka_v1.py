# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for KafkaDataset."""


import time
import pytest

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from tensorflow import dtypes  # pylint: disable=wrong-import-position
from tensorflow import errors  # pylint: disable=wrong-import-position
from tensorflow import test  # pylint: disable=wrong-import-position
from tensorflow.compat.v1 import data  # pylint: disable=wrong-import-position

import tensorflow_io.kafka as kafka_io  # pylint: disable=wrong-import-position


class KafkaDatasetTest(test.TestCase):
    """Tests for KafkaDataset."""

    # The Kafka server has to be setup before the test
    # and tear down after the test manually.
    # The docker engine has to be installed.
    #
    # To setup the Kafka server:
    # $ bash kafka_test.sh start kafka
    #
    # To tear down the Kafka server:
    # $ bash kafka_test.sh stop kafka

    def test_kafka_dataset(self):
        """Tests for KafkaDataset when reading non-keyed messages
        from a single-partitioned topic"""
        topics = tf.compat.v1.placeholder(dtypes.string, shape=[None])
        num_epochs = tf.compat.v1.placeholder(dtypes.int64, shape=[])
        batch_size = tf.compat.v1.placeholder(dtypes.int64, shape=[])

        repeat_dataset = kafka_io.KafkaDataset(topics, group="test", eof=True).repeat(
            num_epochs
        )
        batch_dataset = repeat_dataset.batch(batch_size)

        iterator = data.Iterator.from_structure(batch_dataset.output_types)
        init_op = iterator.make_initializer(repeat_dataset)
        init_batch_op = iterator.make_initializer(batch_dataset)
        get_next = iterator.get_next()

        with self.cached_session() as sess:
            # Basic test: read a limited number of messages from the topic.
            sess.run(init_op, feed_dict={topics: ["test:0:0:4"], num_epochs: 1})
            for i in range(5):
                self.assertEqual(("D" + str(i)).encode(), sess.run(get_next))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

            # Basic test: read all the messages from the topic from offset 5.
            sess.run(init_op, feed_dict={topics: ["test:0:5:-1"], num_epochs: 1})
            for i in range(5):
                self.assertEqual(("D" + str(i + 5)).encode(), sess.run(get_next))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

            # Basic test: read from different subscriptions of the same topic.
            sess.run(
                init_op,
                feed_dict={topics: ["test:0:0:4", "test:0:5:-1"], num_epochs: 1},
            )
            for j in range(2):
                for i in range(5):
                    self.assertEqual(
                        ("D" + str(i + j * 5)).encode(), sess.run(get_next)
                    )
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

            # Test repeated iteration through both subscriptions.
            sess.run(
                init_op,
                feed_dict={topics: ["test:0:0:4", "test:0:5:-1"], num_epochs: 10},
            )
            for _ in range(10):
                for j in range(2):
                    for i in range(5):
                        self.assertEqual(
                            ("D" + str(i + j * 5)).encode(), sess.run(get_next)
                        )
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

            # Test batched and repeated iteration through both subscriptions.
            sess.run(
                init_batch_op,
                feed_dict={
                    topics: ["test:0:0:4", "test:0:5:-1"],
                    num_epochs: 10,
                    batch_size: 5,
                },
            )
            for _ in range(10):
                self.assertAllEqual(
                    [("D" + str(i)).encode() for i in range(5)], sess.run(get_next)
                )
                self.assertAllEqual(
                    [("D" + str(i + 5)).encode() for i in range(5)], sess.run(get_next)
                )

    @pytest.mark.skip(reason="TODO")
    def test_kafka_dataset_save_and_restore(self):
        """Tests for KafkaDataset save and restore."""
        g = tf.Graph()
        with g.as_default():
            topics = tf.compat.v1.placeholder(dtypes.string, shape=[None])
            num_epochs = tf.compat.v1.placeholder(dtypes.int64, shape=[])

            repeat_dataset = kafka_io.KafkaDataset(
                topics, group="test", eof=True
            ).repeat(num_epochs)
            iterator = repeat_dataset.make_initializable_iterator()
            get_next = iterator.get_next()

            it = tf.data.experimental.make_saveable_from_iterator(iterator)
            g.add_to_collection(tf.compat.v1.GraphKeys.SAVEABLE_OBJECTS, it)
            saver = tf.compat.v1.train.Saver()

            model_file = "/tmp/test-kafka-model"
            with self.cached_session() as sess:
                sess.run(
                    iterator.initializer,
                    feed_dict={topics: ["test:0:0:4"], num_epochs: 1},
                )
                for i in range(3):
                    self.assertEqual(("D" + str(i)).encode(), sess.run(get_next))
                # Save current offset which is 2
                saver.save(sess, model_file, global_step=3)

            checkpoint_file = "/tmp/test-kafka-model-3"
            with self.cached_session() as sess:
                saver.restore(sess, checkpoint_file)
                # Restore current offset to 2
                for i in [2, 3]:
                    self.assertEqual(("D" + str(i)).encode(), sess.run(get_next))

    def test_kafka_topic_configuration(self):
        """Tests for KafkaDataset topic configuration properties."""
        topics = tf.compat.v1.placeholder(dtypes.string, shape=[None])
        num_epochs = tf.compat.v1.placeholder(dtypes.int64, shape=[])
        cfg_list = ["auto.offset.reset=earliest"]

        repeat_dataset = kafka_io.KafkaDataset(
            topics, group="test", eof=True, config_topic=cfg_list
        ).repeat(num_epochs)

        iterator = data.Iterator.from_structure(repeat_dataset.output_types)
        init_op = iterator.make_initializer(repeat_dataset)
        get_next = iterator.get_next()

        with self.cached_session() as sess:
            # Use a wrong offset 100 here to make sure
            # configuration 'auto.offset.reset=earliest' works.
            sess.run(init_op, feed_dict={topics: ["test:0:100:-1"], num_epochs: 1})
            for i in range(5):
                self.assertEqual(("D" + str(i)).encode(), sess.run(get_next))

    def test_kafka_global_configuration(self):
        """Tests for KafkaDataset global configuration properties."""
        topics = tf.compat.v1.placeholder(dtypes.string, shape=[None])
        num_epochs = tf.compat.v1.placeholder(dtypes.int64, shape=[])
        cfg_list = ["debug=generic", "enable.auto.commit=false"]

        repeat_dataset = kafka_io.KafkaDataset(
            topics, group="test", eof=True, config_global=cfg_list
        ).repeat(num_epochs)

        iterator = data.Iterator.from_structure(repeat_dataset.output_types)
        init_op = iterator.make_initializer(repeat_dataset)
        get_next = iterator.get_next()

        with self.cached_session() as sess:
            sess.run(init_op, feed_dict={topics: ["test:0:0:4"], num_epochs: 1})
            for i in range(5):
                self.assertEqual(("D" + str(i)).encode(), sess.run(get_next))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

    def test_kafka_wrong_global_configuration_failed(self):
        """Tests for KafkaDataset worng global configuration properties."""
        topics = tf.compat.v1.placeholder(dtypes.string, shape=[None])
        num_epochs = tf.compat.v1.placeholder(dtypes.int64, shape=[])

        # Add wrong configuration
        wrong_cfg = ["debug=al"]
        repeat_dataset = kafka_io.KafkaDataset(
            topics, group="test", eof=True, config_global=wrong_cfg
        ).repeat(num_epochs)

        iterator = data.Iterator.from_structure(repeat_dataset.output_types)
        init_op = iterator.make_initializer(repeat_dataset)
        get_next = iterator.get_next()

        with self.cached_session() as sess:
            sess.run(init_op, feed_dict={topics: ["test:0:0:4"], num_epochs: 1})
            with self.assertRaises(errors.InternalError):
                sess.run(get_next)

    def test_kafka_wrong_topic_configuration_failed(self):
        """Tests for KafkaDataset wrong topic configuration properties."""
        topics = tf.compat.v1.placeholder(dtypes.string, shape=[None])
        num_epochs = tf.compat.v1.placeholder(dtypes.int64, shape=[])

        # Add wrong configuration
        wrong_cfg = ["auto.offset.reset=arliest"]
        repeat_dataset = kafka_io.KafkaDataset(
            topics, group="test", eof=True, config_topic=wrong_cfg
        ).repeat(num_epochs)

        iterator = data.Iterator.from_structure(repeat_dataset.output_types)
        init_op = iterator.make_initializer(repeat_dataset)
        get_next = iterator.get_next()

        with self.cached_session() as sess:
            sess.run(init_op, feed_dict={topics: ["test:0:0:4"], num_epochs: 1})
            with self.assertRaises(errors.InternalError):
                sess.run(get_next)

    @pytest.mark.skip(reason="TODO")
    def test_write_kafka(self):
        """test_write_kafka"""
        channel = "e{}e".format(time.time())

        # Start with reading test topic, replace `D` with `e(time)e`,
        # and write to test_e(time)e` topic.
        dataset = kafka_io.KafkaDataset(topics=["test:0:0:4"], group="test", eof=True)
        dataset = dataset.map(
            lambda x: kafka_io.write_kafka(
                tf.strings.regex_replace(x, "D", channel), topic="test_" + channel
            )
        )
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        get_next = iterator.get_next()

        with self.cached_session() as sess:
            # Basic test: read from topic 0.
            sess.run(init_op)
            for i in range(5):
                self.assertEqual((channel + str(i)).encode(), sess.run(get_next))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

        # Reading from `test_e(time)e` we should get the same result
        dataset = kafka_io.KafkaDataset(
            topics=["test_" + channel], group="test", eof=True
        )
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        get_next = iterator.get_next()

        with self.cached_session() as sess:
            sess.run(init_op)
            for i in range(5):
                self.assertEqual((channel + str(i)).encode(), sess.run(get_next))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

    def test_kafka_dataset_with_key(self):
        """Tests for KafkaDataset when reading keyed-messages
        from a single-partitioned topic"""
        topics = tf.compat.v1.placeholder(dtypes.string, shape=[None])
        num_epochs = tf.compat.v1.placeholder(dtypes.int64, shape=[])
        batch_size = tf.compat.v1.placeholder(dtypes.int64, shape=[])

        repeat_dataset = kafka_io.KafkaDataset(
            topics, group="test", eof=True, message_key=True
        ).repeat(num_epochs)
        batch_dataset = repeat_dataset.batch(batch_size)

        iterator = data.Iterator.from_structure(batch_dataset.output_types)
        init_op = iterator.make_initializer(repeat_dataset)
        init_batch_op = iterator.make_initializer(batch_dataset)
        get_next = iterator.get_next()

        with self.cached_session() as sess:
            # Basic test: read a limited number of keyed messages from the topic.
            sess.run(init_op, feed_dict={topics: ["key-test:0:0:4"], num_epochs: 1})
            for i in range(5):
                self.assertEqual(
                    (("D" + str(i)).encode(), ("K" + str(i % 2)).encode()),
                    sess.run(get_next),
                )
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

            # Basic test: read all the keyed messages from the topic from offset 5.
            sess.run(init_op, feed_dict={topics: ["key-test:0:5:-1"], num_epochs: 1})
            for i in range(5):
                self.assertEqual(
                    (("D" + str(i + 5)).encode(), ("K" + str((i + 5) % 2)).encode()),
                    sess.run(get_next),
                )
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

            # Basic test: read from different subscriptions of the same topic.
            sess.run(
                init_op,
                feed_dict={
                    topics: ["key-test:0:0:4", "key-test:0:5:-1"],
                    num_epochs: 1,
                },
            )
            for j in range(2):
                for i in range(5):
                    self.assertEqual(
                        (
                            ("D" + str(i + j * 5)).encode(),
                            ("K" + str((i + j * 5) % 2)).encode(),
                        ),
                        sess.run(get_next),
                    )
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

            # Test repeated iteration through both subscriptions.
            sess.run(
                init_op,
                feed_dict={
                    topics: ["key-test:0:0:4", "key-test:0:5:-1"],
                    num_epochs: 10,
                },
            )
            for _ in range(10):
                for j in range(2):
                    for i in range(5):
                        self.assertEqual(
                            (
                                ("D" + str(i + j * 5)).encode(),
                                ("K" + str((i + j * 5) % 2)).encode(),
                            ),
                            sess.run(get_next),
                        )
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

            # Test batched and repeated iteration through both subscriptions.
            sess.run(
                init_batch_op,
                feed_dict={
                    topics: ["key-test:0:0:4", "key-test:0:5:-1"],
                    num_epochs: 10,
                    batch_size: 5,
                },
            )
            for _ in range(10):
                self.assertAllEqual(
                    [
                        [("D" + str(i)).encode() for i in range(5)],
                        [("K" + str(i % 2)).encode() for i in range(5)],
                    ],
                    sess.run(get_next),
                )
                self.assertAllEqual(
                    [
                        [("D" + str(i + 5)).encode() for i in range(5)],
                        [("K" + str((i + 5) % 2)).encode() for i in range(5)],
                    ],
                    sess.run(get_next),
                )

    def test_kafka_dataset_with_partitioned_key(self):
        """Tests for KafkaDataset when reading keyed-messages
        from a multi-partitioned topic"""
        topics = tf.compat.v1.placeholder(dtypes.string, shape=[None])
        num_epochs = tf.compat.v1.placeholder(dtypes.int64, shape=[])
        batch_size = tf.compat.v1.placeholder(dtypes.int64, shape=[])

        repeat_dataset = kafka_io.KafkaDataset(
            topics, group="test", eof=True, message_key=True
        ).repeat(num_epochs)
        batch_dataset = repeat_dataset.batch(batch_size)

        iterator = data.Iterator.from_structure(batch_dataset.output_types)
        init_op = iterator.make_initializer(repeat_dataset)
        init_batch_op = iterator.make_initializer(batch_dataset)
        get_next = iterator.get_next()

        with self.cached_session() as sess:
            # Basic test: read first 5 messages from the first partition of the topic.
            # NOTE: The key-partition mapping occurs based on the order in which the data
            # is being stored in kafka. Please check kafka_test.sh for the sample data.

            sess.run(
                init_op,
                feed_dict={topics: ["key-partition-test:0:0:5"], num_epochs: 1},
            )
            for i in range(5):
                self.assertEqual(
                    (("D" + str(i * 2)).encode(), (b"K0")), sess.run(get_next),
                )
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

            # Basic test: read first 5 messages from the second partition of the topic.
            sess.run(
                init_op,
                feed_dict={topics: ["key-partition-test:1:0:5"], num_epochs: 1},
            )
            for i in range(5):
                self.assertEqual(
                    (("D" + str(i * 2 + 1)).encode(), (b"K1")), sess.run(get_next),
                )
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

            # Basic test: read from different subscriptions to the same topic.
            sess.run(
                init_op,
                feed_dict={
                    topics: ["key-partition-test:0:0:5", "key-partition-test:1:0:5"],
                    num_epochs: 1,
                },
            )
            for j in range(2):
                for i in range(5):
                    self.assertEqual(
                        (("D" + str(i * 2 + j)).encode(), ("K" + str(j)).encode()),
                        sess.run(get_next),
                    )
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

            # Test repeated iteration through both subscriptions.
            sess.run(
                init_op,
                feed_dict={
                    topics: ["key-partition-test:0:0:5", "key-partition-test:1:0:5"],
                    num_epochs: 10,
                },
            )
            for _ in range(10):
                for j in range(2):
                    for i in range(5):
                        self.assertEqual(
                            (("D" + str(i * 2 + j)).encode(), ("K" + str(j)).encode()),
                            sess.run(get_next),
                        )
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

            # Test batched and repeated iteration through both subscriptions.
            sess.run(
                init_batch_op,
                feed_dict={
                    topics: ["key-partition-test:0:0:5", "key-partition-test:1:0:5"],
                    num_epochs: 10,
                    batch_size: 5,
                },
            )
            for _ in range(10):
                for j in range(2):
                    self.assertAllEqual(
                        [
                            [("D" + str(i * 2 + j)).encode() for i in range(5)],
                            [("K" + str(j)).encode() for i in range(5)],
                        ],
                        sess.run(get_next),
                    )

    def test_kafka_dataset_with_offset(self):
        """Tests for KafkaDataset when reading non-keyed messages
        from a single-partitioned topic"""
        topics = tf.compat.v1.placeholder(dtypes.string, shape=[None])
        num_epochs = tf.compat.v1.placeholder(dtypes.int64, shape=[])
        batch_size = tf.compat.v1.placeholder(dtypes.int64, shape=[])

        repeat_dataset = kafka_io.KafkaDataset(
            topics, group="test", eof=True, message_offset=True
        ).repeat(num_epochs)
        batch_dataset = repeat_dataset.batch(batch_size)

        iterator = data.Iterator.from_structure(batch_dataset.output_types)
        init_op = iterator.make_initializer(repeat_dataset)
        get_next = iterator.get_next()

        with self.cached_session() as sess:
            # Basic offset test: read a limited number of messages from the topic.
            sess.run(init_op, feed_dict={topics: ["offset-test:0:0:4"], num_epochs: 1})
            for i in range(5):
                self.assertEqual(
                    (("D" + str(i)).encode(), ("0:" + str(i)).encode()),
                    sess.run(get_next),
                )
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)


if __name__ == "__main__":
    test.main()
