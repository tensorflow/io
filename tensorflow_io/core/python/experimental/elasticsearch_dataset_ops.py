# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""ElasticsearchIODatasets"""

from urllib.parse import urlparse
import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.experimental import serialization_ops


class _ElasticsearchHandler:
    """Utility class to facilitate API queries and state management of
        session data.
    """

    def __init__(self, nodes, index, doc_type, headers_dict):
        self.nodes = nodes
        self.index = index
        self.doc_type = doc_type
        self.headers_dict = headers_dict
        self.prepare_base_urls()
        self.prepare_connection_data()

    def prepare_base_urls(self):
        """Prepares the base url for establish connection with the
        elasticsearch master.

        Returns:
            A list of base_url's, each of type tf.string for establishing
            the connection pool.
        """

        if self.nodes is None:
            self.nodes = ["http://localhost:9200"]

        elif isinstance(self.nodes, str):
            self.nodes = [self.nodes]

        self.base_urls = []
        for node in self.nodes:
            if "//" not in node:
                raise ValueError(
                    "Please provide the list of nodes in 'protocol://host:port' format."
                )

            # Additional sanity check
            url_obj = urlparse(node)
            base_url = "{}://{}".format(url_obj.scheme, url_obj.netloc)
            self.base_urls.append(base_url)

    def prepare_connection_data(self):
        """Prepares the healthcheck and resource urls from the base_urls"""

        self.healthcheck_urls = [
            "{}/_cluster/health".format(base_url) for base_url in self.base_urls
        ]
        self.request_urls = []
        for base_url in self.base_urls:
            if self.doc_type is None:
                request_url = "{}/{}/_search?scroll=1m".format(base_url, self.index)
            else:
                request_url = "{}/{}/{}/_search?scroll=1m".format(
                    base_url, self.index, self.doc_type
                )
            self.request_urls.append(request_url)

        self.headers = ["Content-Type=application/json"]
        if self.headers_dict is not None:
            if isinstance(self.headers_dict, dict):
                for key, value in self.headers_dict.items():
                    if key.lower() == "content-type":
                        continue
                    self.headers.append("{}={}".format(key, value))
            else:
                raise ValueError(
                    "Headers should be a dict of key:value pairs. Got: ", self.headers
                )

    def get_healthy_resource(self):
        """Retrieve the resource which is connected to a healthy node"""

        for healthcheck_url, request_url in zip(
            self.healthcheck_urls, self.request_urls
        ):
            try:
                resource, columns, raw_dtypes = core_ops.io_elasticsearch_readable_init(
                    healthcheck_url=healthcheck_url,
                    healthcheck_field="status",
                    request_url=request_url,
                    headers=self.headers,
                )
                print("Connection successful: {}".format(healthcheck_url))
                dtypes = []
                for dtype in raw_dtypes:
                    if dtype == "DT_INT32":
                        dtypes.append(tf.int32)
                    elif dtype == "DT_INT64":
                        dtypes.append(tf.int64)
                    elif dtype == "DT_DOUBLE":
                        dtypes.append(tf.double)
                    elif dtype == "DT_STRING":
                        dtypes.append(tf.string)
                    elif dtype == "DT_BOOL":
                        dtypes.append(tf.bool)
                return resource, columns.numpy(), dtypes, request_url
            except Exception:
                print("Skipping node: {}".format(healthcheck_url))
                continue
        else:
            raise ConnectionError(
                "No healthy node available for the index: {}, please check the cluster config".format(
                    self.index
                )
            )

    def get_next_batch(self, resource, request_url):
        """Prepares the next batch of data based on the request url and
        the counter index.

        Args:
            resource: the init op resource.
            columns: list of columns to prepare the structured data.
            dtypes: tf.dtypes of the columns.
            request_url: The request url to fetch the data
        Returns:
            A Tensor containing serialized JSON records.
        """

        url_obj = urlparse(request_url)
        scroll_request_url = "{}://{}/_search/scroll".format(
            url_obj.scheme, url_obj.netloc
        )

        values = core_ops.io_elasticsearch_readable_next(
            resource=resource,
            request_url=request_url,
            scroll_request_url=scroll_request_url,
        )
        return values

    def parse_json(self, raw_item, columns, dtypes):
        """Prepares the next batch of data based on the request url and
        the counter index.

        Args:
            raw_value: A serialized JSON record in tf.string format.
            columns: list of columns to prepare the structured data.
            dtypes: tf.dtypes of the columns.
        Returns:
            Structured data with columns as keys and the corresponding tensors as values.
        """
        specs = {}
        for column, dtype in zip(columns, dtypes):
            specs[column.decode("utf-8")] = tf.TensorSpec(tf.TensorShape([]), dtype)
        parsed_item = serialization_ops.decode_json(data=raw_item, specs=specs)
        return parsed_item


class ElasticsearchIODataset(tf.compat.v2.data.Dataset):
    """Represents an elasticsearch based tf.data.Dataset

    The records fetched from the cluster are structured in their content and
    require additional processing to make them ready for training the
    machine learning model.

    There are various ways of converting column data into features and using them
    to train the models. For example, let's consider an elasticsearch dataset
    which contains records having the `fare`, `age` and `survived` keys. The
    values of the `survived` key act as our label data.

    >>> import tensorflow as tf
    >>> from tensorflow import feature_column
    >>> from tensorflow.keras import layers
    >>> import tensorflow_io as tfio

    >>> dataset = tfio.experimental.elasticsearch.ElasticsearchIODataset(
                    nodes=["localhost:9092"],
                    index="people",
                    doc_type="survivors")
    >>> dataset = dataset.map(lambda v: (v, v.pop("survived")))
    >>> dataset = dataset.batch(10)

    >>> fare = feature_column.numeric_column('fare') # numeric column
    >>> age = feature_column.numeric_column('age')
    >>> age_buckets = feature_column.bucketized_column(age) # bucketized column

    >>> feature_columns = [cost, age_buckets]
    >>> feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    The `feature_layer` can now be added as the input layer to the `tf.keras` model.

    >>> model = tf.keras.Sequential([
            feature_layer,
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(1),
        ])
    >>> model.compile(optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    >>> model.fit(dataset, epochs=5)

    Additionally, while creating the `ElasticsearchIODataset`, headers can be passed
    to connect to clusters that require additional configuration. For example, passing
    the authorization headers:

    >>> HEADERS = {"Authorization": "Basic ZWxhc3RpYzpkZWZhdWx0X3Bhc3N3b3Jk"}
    >>> dataset = tfio.experimental.elasticsearch.ElasticsearchIODataset(
                    nodes=["localhost:9092"],
                    index="people",
                    doc_type="survivors",
                    headers=HEADERS)
    """

    def __init__(self, nodes, index, doc_type=None, headers=None, internal=True):
        """Prepare the ElasticsearchIODataset.

        Args:
            nodes: A `tf.string` tensor containing the hostnames of nodes
                in [protocol://hostname:port] format.
                For example: ["http://localhost:9200"]
            index: A `tf.string` representing the elasticsearch index to query.
            doc_type: (Optional) A `tf.string` representing the type of documents
                in the index to query.
            headers: (Optional) A dict of headers. For example:
                {'Content-Type': 'application/json'}
        """
        with tf.name_scope("ElasticsearchIODataset"):
            assert internal

            handler = _ElasticsearchHandler(
                nodes=nodes, index=index, doc_type=doc_type, headers_dict=headers
            )
            resource, columns, dtypes, request_url = handler.get_healthy_resource()

            dataset = tf.data.experimental.Counter()
            dataset = dataset.map(
                lambda i: handler.get_next_batch(
                    resource=resource, request_url=request_url
                )
            )
            dataset = dataset.apply(
                tf.data.experimental.take_while(lambda v: tf.greater(tf.shape(v)[0], 0))
            )
            dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
            dataset = dataset.map(
                lambda v: handler.parse_json(v, columns=columns, dtypes=dtypes),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            self._dataset = dataset

            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
