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
"""MongoDBWriter"""

import json
from urllib.parse import urlparse
import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.experimental import serialization_ops


class MongoDBWriter:
    """Write documents to mongoDB.

    The writer can be used to store documents in mongoDB while dealing with tensorflow
    based models and inference outputs. Without loss of generality, consider an ML
    model that is being used for inference. The outputs of inference can be modelled into
    a structured record by enriching the schema with additional information( for ex: metadata
    about input data and the semantics of the inference etc.) and can be stored in mongo
    collections for persistence or future analysis.

    To make a connection and write the documents to the mongo collections,
    the `tfio.experimental.mongodb.MongoDBWriter` API can be used.

    Example:

    >>> URI = "mongodb://mongoadmin:default_password@localhost:27017"
    >>> DATABASE = "tfiodb"
    >>> COLLECTION = "test"
    >>> writer = tfio.experimental.mongodb.MongoDBWriter(
        uri=URI, database=DATABASE, collection=COLLECTION
    )
    >>> for i in range(1000):
    ...    data = {"key{}".format(i): "value{}".format(i)}
    ...    writer.write(data)

    """

    def __init__(self, uri, database, collection):
        """Initialize the dataset with the following parameters

        Args:
            uri: The uri of the mongo server or replicaset to connect to.
                - To connect to a MongoDB server with username and password
                based authentication, the following uri pattern can be used.
                Example: `"mongodb://mongoadmin:default_password@localhost:27017"`.

                - Connecting to a replica set is much like connecting to a
                standalone MongoDB server. Simply specify the replica set name
                using the `?replicaSet=myreplset` URI option.
                Example: "mongodb://host01:27017,host02:27017,host03:27017/?replicaSet=myreplset"

                Additional information on writing uri's can be found here:
                - [libmongoc uri docs](http://mongoc.org/libmongoc/current/mongoc_uri_t.html)
                - [mongodb uri docs](https://docs.mongodb.com/manual/reference/connection-string/)
            database: The database in the standalone standalone MongoDB server or a replica set
                to connect to.
            collection: The collection from which the documents have to be retrieved.
        """
        self.uri = uri
        self.database = database
        self.collection = collection
        self.resource = core_ops.io_mongo_db_writable_init(
            uri=self.uri, database=self.database, collection=self.collection,
        )

    def write(self, doc):
        """Insert a single json document"""

        core_ops.io_mongo_db_writable_write(
            resource=self.resource, record=json.dumps(doc)
        )

    def _delete_many(self, doc):
        """Delete all matching documents"""

        core_ops.io_mongo_db_writable_delete_many(
            resource=self.resource, record=json.dumps(doc)
        )
