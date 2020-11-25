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

URI = "mongodb://mongoadmin@default_password@localhost:27017"
DATABASE = "tfiodb"
COLLECTION = "test"


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
def test_ping():
    """Test the resource creation"""

    handler = tfio.experimental.mongodb._MongoDBHandler(
        uri=URI, database=DATABASE, collection=COLLECTION
    )
