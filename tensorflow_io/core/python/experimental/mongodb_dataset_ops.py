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
"""MongoDBIODatasets"""

from urllib.parse import urlparse
import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.experimental import serialization_ops


class _MongoDBHandler:
    """Utility class to facilitate API queries and state management of
    session data.
    """

    def __init__(self, uri, database, collection):
        self.uri = uri
        self.database = database
        self.collection = collection
        self.get_healthy_resource()

    def get_healthy_resource(self):
        """Retrieve the resource which is connected to a healthy node"""

        resource = core_ops.io_mongo_db_readable_init(
            uri=self.uri, database=self.database, collection=self.collection,
        )
        print("Connection successful: {}".format(self.uri))
        return resource

    def get_next_batch(self, resource):
        """Prepares the next batch of data based on the request url and
        the counter index.

        Args:
            resource: the init op resource.
        Returns:
            A Tensor containing serialized JSON records.
        """

        return core_ops.io_mongo_db_readable_next(resource=resource)


class MongoDBIODataset(tf.data.Dataset):
    """Fetch records from mongoDB"""

    def __init__(self, uri, database, collection):
        handler = _MongoDBHandler(uri=uri, database=database, collection=collection)
        resource = handler.get_healthy_resource()
        dataset = tf.data.experimental.Counter()
        dataset = dataset.map(lambda i: handler.get_next_batch(resource=resource))
        dataset = dataset.apply(
            tf.data.experimental.take_while(lambda v: tf.greater(tf.shape(v)[0], 0))
        )
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        self._dataset = dataset

        super().__init__(
            self._dataset._variant_tensor
        )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
