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
from tensorflow_io.core.python.ops import data_ops as data_ops
from tensorflow_io.core.python.ops import core_ops as prometheus_ops

class PrometheusDataset(data_ops.Dataset):
  """A Prometheus Dataset
  """

  def __init__(self, endpoint, schema=None, batch=None):
    """Create a Prometheus Reader.

    Args:
      endpoint: A `tf.string` tensor containing address of
        the prometheus server.
      schema: A `tf.string` tensor containing the query
        string.
      batch: Size of the batch.
    """
    batch = 0 if batch is None else batch
    dtypes = [tf.int64, tf.float64]
    shapes = [
        tf.TensorShape([]), tensorflow.TensorShape([])] if batch == 0 else [
            tf.TensorShape([None]), tf.TensorShape([None])]
    super(PrometheusDataset, self).__init__(
        prometheus_ops.prometheus_dataset,
        prometheus_ops.prometheus_input(endpoint, schema=schema),
        batch, dtypes, shapes)
