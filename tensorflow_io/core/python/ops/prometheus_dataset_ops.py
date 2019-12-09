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
"""PrometheusDataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.data.experimental.ops.sleep import sleep
from tensorflow_io.core.python.ops import golang_ops

class PrometheusGraphIODataset(tf.data.Dataset):
  """PrometheusGraphIODataset"""

  def __init__(self,
               query,
               length,
               offset=None,
               endpoint=None,
               internal=True):
    """PrometheusGraphIODataset."""
    with tf.name_scope("PrometheusGraphIODataset"):
      assert internal

      metadata = []
      metadata.append("length=%d" % length)
      if offset is not None:
        metadata.append("offset=%d" % offset)
      if endpoint is not None:
        metadata.append("endpoint=%d" % endpoint)
      resource = golang_ops.io_prometheus_readable_init(query, metadata)

      step = 1 * 1000 # 1 second

      self._resource = resource
      start, stop = golang_ops.io_prometheus_readable_spec(resource)
      indices_start = tf.data.Dataset.range(start, stop, step)
      indices_stop = indices_start.skip(1).concatenate(
          tf.data.Dataset.from_tensor_slices([stop]))
      dataset = tf.data.Dataset.zip((indices_start, indices_stop))
      dataset = dataset.map(
          lambda start, stop: golang_ops.io_prometheus_readable_read(
              resource, start, stop))
      dataset = dataset.unbatch()
      self._dataset = dataset
      super(PrometheusGraphIODataset, self).__init__(
          self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec

class PrometheusScrapeStreamIODataset(tf.data.Dataset):
  """PrometheusScrapeStreamGraphIODataset"""

  def __init__(self,
               metric,
               endpoint,
               interval=None,
               internal=True):
    """PrometheusScrapeStreamIODataset."""
    with tf.name_scope("PrometheusScrapeStreamIODataset"):
      assert internal

      interval = 1000000 if interval is None else interval

      dataset = tf.data.Dataset.range(0, 10, 1)
      dataset = dataset.map(lambda i: golang_ops.io_prometheus_scrape(metric, endpoint, i))
      dataset = dataset.apply(sleep(interval))

      self._dataset = dataset
      super(PrometheusScrapeStreamIODataset, self).__init__(
          self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec
