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
import tensorflow_io as tfio

# COMMON VARIABLES

ES_CONTAINER_NAME = "tfio-elasticsearch"
NODE = "http://localhost:9200"
INDEX = "people"
DOC_TYPE = "survivors"
HEADERS = {"Content-Type": "application/json"}
ATTRS = ["name", "gender", "age", "fare", "survived"]


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

    create_index_url = "{}/{}".format(NODE, INDEX)
    res = requests.put(create_index_url)
    assert res.status_code == 200


@pytest.mark.parametrize(
    "record",
    [
        (("person1", "Male", 20, 80.52, 1)),
        (("person2", "Female", 30, 40.88, 0)),
        (("person3", "Male", 40, 20.73, 0)),
        (("person4", "Female", 50, 100.99, 1)),
    ],
)
@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_populate_data(record):
    """Populate the index with data"""

    put_item_url = "{}/{}/{}".format(NODE, INDEX, DOC_TYPE)
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
        nodes=[NODE], index=INDEX, doc_type=DOC_TYPE
    )

    assert issubclass(type(dataset), tf.data.Dataset)

    for item in dataset:
        for attr in ATTRS:
            assert attr in item


@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_elasticsearch_io_dataset_batch():
    """Test the functionality of the ElasticsearchIODataset"""

    BATCH_SIZE = 2
    dataset = tfio.experimental.elasticsearch.ElasticsearchIODataset(
        nodes=[NODE], index=INDEX, doc_type=DOC_TYPE
    ).batch(BATCH_SIZE)

    assert issubclass(type(dataset), tf.data.Dataset)

    for item in dataset:
        for attr in ATTRS:
            assert attr in item
            assert len(item[attr]) == BATCH_SIZE


@pytest.mark.skipif(not is_container_running(), reason="The container is not running")
def test_cleanup():
    """Clean up the index"""

    delete_index_url = "{}/{}".format(NODE, INDEX)
    res = requests.delete(delete_index_url)
    assert res.status_code == 200
