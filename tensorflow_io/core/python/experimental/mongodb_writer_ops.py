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
    """Write documents to mongoDB"""

    def __init__(self, uri, database, collection):

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
