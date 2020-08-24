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


class ElasticsearchIODataset(tf.data.Dataset):
    """Represents an elasticsearch based tf.data.Dataset"""

    def __init__(self, hosts=None, protocol="http", internal=True):
        with tf.name_scope("ElasticsearchIODataset"):
            assert internal

            base_urls = prepare_base_urls(hosts=hosts, protocol=protocol)
            healthcheck_urls = prepare_healthcheck_urls(base_urls=base_urls)
            resource = get_healthy_resource(healthcheck_urls=healthcheck_urls)

            # TODO (kvignesh1420): Read data from the cluster

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec


def prepare_base_urls(hosts, protocol):
    """Prepares the base url for establish connection with the elasticsearch master
    
    Args:
        hosts: A list of tf.strings in the host:port format.
        protocol: The protocol to be used to connect to the elasticsearch cluster.
    Returns:
        A list of base_url's, each of type tf.string for establishing the connection pool
    """

    if hosts is None:
        hosts = ["localhost:9200"]

    elif isinstance(hosts, str):
        hosts = [hosts]

    base_urls = []
    for host in hosts:
        if "//" in host:
            raise ValueError("Please provide the list of hosts in 'host:port' format.")

        base_url = "{}://{}".format(protocol, host)
        base_urls.append(base_url)

    return base_urls


def prepare_healthcheck_urls(base_urls):
    """Appends the healthcheck path to all the base_urls"""

    return ["{}/_cluster/health".format(base_url) for base_url in base_urls]


def get_healthy_resource(healthcheck_urls):
    """Retrieve the resource which is connected to a healthy node"""

    for url in healthcheck_urls:
        try:
            resource = core_ops.io_elasticsearch_readable_init(
                url=url, healthcheck_field="status"
            )
            print("Connection successful:{}".format(url))
            return resource
        except:
            print("Skipping host:{}".format(url))
            continue
    else:
        raise Exception("No healthy node available, please check your cluster status")
