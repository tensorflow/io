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
"""PrometheusDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops
from tensorflow_io.core.python.ops import core_ops

def read_prometheus(endpoint, query):
  """read_prometheus"""
  return core_ops.read_prometheus(endpoint, query)

class PrometheusDataset(data_ops.BaseDataset):
  """A Prometheus Dataset"""

  def __init__(self, endpoint, query):
    """Create a Prometheus Dataset

    Args:
      endpoint: A `tf.string` tensor containing address of
        the prometheus server.
      query: A `tf.string` tensor containing the query
        string.
    """
    dtypes = [tf.int64, tf.float64]
    shapes = [tf.TensorShape([None]), tf.TensorShape([None])]
    # TODO: It could be possible to improve the performance
    # by reading a small chunk of the data while at the same
    # time allowing reuse of read_prometheus. Essentially
    # read_prometheus could take a timestamp and read small chunk
    # at a time until running out of data.
    timestamp, value = read_prometheus(endpoint, query)
    timestamp_dataset = data_ops.BaseDataset.from_tensors(timestamp)
    value_dataset = data_ops.BaseDataset.from_tensors(value)
    dataset = data_ops.BaseDataset.zip((timestamp_dataset, value_dataset))

    self._dataset = dataset
    super(PrometheusDataset, self).__init__(
        self._dataset._variant_tensor, dtypes, shapes) # pylint: disable=protected-access
