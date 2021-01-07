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

"""Tests for the mongodb datasets"""

import socket
import pytest
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
import tensorflow_io as tfio

# COMMON VARIABLES
URI = "mongodb://mongoadmin:default_password@localhost:27017"
DATABASE = "tfiodb"
COLLECTION = "test"
RECORDS = [
    {
        "name": "person1",
        "gender": "Male",
        "age": 20,
        "fare": 80.52,
        "vip": False,
        "survived": 1,
    },
    {
        "name": "person2",
        "gender": "Female",
        "age": 20,
        "fare": 40.88,
        "vip": True,
        "survived": 0,
    },
] * 1000
SPECS = {
    "name": tf.TensorSpec(tf.TensorShape([]), tf.string),
    "gender": tf.TensorSpec(tf.TensorShape([]), tf.string),
    "age": tf.TensorSpec(tf.TensorShape([]), tf.int32),
    "fare": tf.TensorSpec(tf.TensorShape([]), tf.float64),
    "vip": tf.TensorSpec(tf.TensorShape([]), tf.bool),
    "survived": tf.TensorSpec(tf.TensorShape([]), tf.int64),
}
BATCH_SIZE = 32


def is_container_running():
    """Check whether the elasticsearch container is up and running
    with the correct port being exposed.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    status = sock.connect_ex(("127.0.0.1", 27017))
    if status == 0:
        return True
    else:
        return False


@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_writer_write():
    """Test the writer resource creation and write operations"""

    writer = tfio.experimental.mongodb.MongoDBWriter(
        uri=URI, database=DATABASE, collection=COLLECTION
    )

    for record in RECORDS:
        writer.write(record)


@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_dataset_read():
    """Test the database creation and read operations"""

    dataset = tfio.experimental.mongodb.MongoDBIODataset(
        uri=URI, database=DATABASE, collection=COLLECTION
    )
    count = 0
    for d in dataset:
        count += 1
    assert count == len(RECORDS)


@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_train_model():
    """Test the dataset by training a tf.keras model"""

    dataset = tfio.experimental.mongodb.MongoDBIODataset(
        uri=URI, database=DATABASE, collection=COLLECTION
    )
    dataset = dataset.map(
        lambda x: tfio.experimental.serialization.decode_json(x, specs=SPECS)
    )
    dataset = dataset.map(lambda v: (v, v.pop("survived")))
    dataset = dataset.batch(BATCH_SIZE)

    assert issubclass(type(dataset), tf.data.Dataset)

    feature_columns = []

    # Numeric column
    fare_column = feature_column.numeric_column("fare")
    feature_columns.append(fare_column)

    # Bucketized column
    age = feature_column.numeric_column("age")
    age_buckets = feature_column.bucketized_column(age, boundaries=[10, 30])
    feature_columns.append(age_buckets)

    # Categorical column
    gender = feature_column.categorical_column_with_vocabulary_list(
        "gender", ["Male", "Female"]
    )
    gender_indicator = feature_column.indicator_column(gender)
    feature_columns.append(gender_indicator)

    # Convert the feature columns into a tf.keras layer
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    # Build the model
    model = tf.keras.Sequential(
        [
            feature_layer,
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(1),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # train the model
    model.fit(dataset, epochs=5)


@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_writer_delete_many():
    """Test the writer resource creation and write operations"""

    writer = tfio.experimental.mongodb.MongoDBWriter(
        uri=URI, database=DATABASE, collection=COLLECTION
    )
    writer._delete_many({})


@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_dataset_read_after_delete():
    """Test the database creation and read operations"""

    dataset = tfio.experimental.mongodb.MongoDBIODataset(
        uri=URI, database=DATABASE, collection=COLLECTION
    )
    count = 0
    for d in dataset:
        count += 1
    assert count == 0
