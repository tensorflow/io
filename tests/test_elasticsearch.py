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

"""Tests for the elasticsearch datasets"""

import time
import json
import pytest
import socket
import requests
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
import tensorflow_io as tfio

# COMMON VARIABLES

ES_CONTAINER_NAME = "tfio-elasticsearch"
NODE = "http://localhost:9200"
INDEX = "people"
DOC_TYPE = "survivors"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Basic ZWxhc3RpYzpkZWZhdWx0X3Bhc3N3b3Jk",
}
ATTRS = ["name", "gender", "age", "fare", "vip", "survived"]


def is_container_running():
    """Check whether the elasticsearch container is up and running
    with the correct port being exposed.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    status = sock.connect_ex(("127.0.0.1", 9200))
    if status == 0:
        return True
    else:
        return False


@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_create_index():
    """Create an index in the cluster"""

    create_index_url = f"{NODE}/{INDEX}"
    res = requests.put(create_index_url, headers=HEADERS)
    assert res.status_code == 200


@pytest.mark.parametrize(
    "record",
    [
        (("person1", "Male", 20, 80.52, False, 1)),
        (("person2", "Female", 30, 40.88, True, 0)),
        (("person3", "Male", 40, 20.73, True, 0)),
        (("person4", "Female", 50, 100.99, False, 1)),
    ],
)
@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_populate_data(record):
    """Populate the index with data"""

    put_item_url = f"{NODE}/{INDEX}/{DOC_TYPE}"
    data = {}
    for idx, attr in enumerate(ATTRS):
        data[attr] = record[idx]

    res = requests.post(put_item_url, json=data, headers=HEADERS)

    # The 201 status code indicates the documents have been properly indexed
    assert res.status_code == 201

    # allow the cluster to index in the background.
    time.sleep(1)


@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_elasticsearch_io_dataset():
    """Test the functionality of the ElasticsearchIODataset"""

    dataset = tfio.experimental.elasticsearch.ElasticsearchIODataset(
        nodes=[NODE], index=INDEX, doc_type=DOC_TYPE, headers=HEADERS
    )

    assert issubclass(type(dataset), tf.data.Dataset)

    for item in dataset:
        for attr in ATTRS:
            assert attr in item


@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_elasticsearch_io_dataset_no_auth():
    """Test the functionality of the ElasticsearchIODataset when basic auth is
    required but the associated header is not passed.
    """

    try:
        dataset = tfio.experimental.elasticsearch.ElasticsearchIODataset(
            nodes=[NODE], index=INDEX, doc_type=DOC_TYPE
        )
    except ConnectionError as e:
        assert str(
            e
        ) == "No healthy node available for the index: {}, please check the cluster config".format(
            INDEX
        )


@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_elasticsearch_io_dataset_batch():
    """Test the functionality of the ElasticsearchIODataset"""

    BATCH_SIZE = 2
    dataset = tfio.experimental.elasticsearch.ElasticsearchIODataset(
        nodes=[NODE], index=INDEX, doc_type=DOC_TYPE, headers=HEADERS
    ).batch(BATCH_SIZE)

    assert issubclass(type(dataset), tf.data.Dataset)

    for item in dataset:
        for attr in ATTRS:
            assert attr in item
            assert len(item[attr]) == BATCH_SIZE


@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_elasticsearch_io_dataset_training():
    """Test the functionality of the ElasticsearchIODataset by training a
    tf.keras model on the structured data.
    """

    BATCH_SIZE = 2
    dataset = tfio.experimental.elasticsearch.ElasticsearchIODataset(
        nodes=[NODE], index=INDEX, doc_type=DOC_TYPE, headers=HEADERS
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
def test_cleanup():
    """Clean up the index"""

    delete_index_url = f"{NODE}/{INDEX}"
    res = requests.delete(delete_index_url, headers=HEADERS)
    assert res.status_code == 200
