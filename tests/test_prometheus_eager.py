# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for PrometheusDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import subprocess
import sys
import pytest
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import tensorflow_io as tfio # pylint: disable=wrong-import-position

if sys.platform == "darwin":
  pytest.skip(
      "prometheus is not supported on macOS yet", allow_module_level=True)

def test_prometheus():
  """test_prometheus"""
  for _ in range(6):
    subprocess.call(["dig", "@localhost", "-p", "1053", "www.google.com"])
    time.sleep(1)
  time.sleep(2)
  prometheus = tfio.IOTensor.from_prometheus(
      "coredns_dns_request_count_total[5s]")
  assert prometheus.index.shape == [5]
  assert prometheus.index.dtype == tf.int64
  assert prometheus.value.shape == [5, 1]
  assert prometheus.value.dtype == tf.float64
  # last value should be 6.0
  assert prometheus.value.to_tensor().numpy()[4] == 6.0

  # test with sklearn.preprocessing, and expect the same as with numpy
  numpy_data = prometheus.value.to_tensor().numpy()

  prometheus_scaler = MinMaxScaler()
  prometheus_scaler.fit(prometheus.value)
  prometheus_transformed = prometheus_scaler.transform(prometheus.value)

  numpy_scaler = MinMaxScaler()
  numpy_scaler.fit(numpy_data)
  numpy_transformed = numpy_scaler.transform(numpy_data)

  assert prometheus_scaler.data_max_ == numpy_scaler.data_max_
  assert prometheus_scaler.data_min_ == numpy_scaler.data_min_
  assert np.all(prometheus_transformed == numpy_transformed)

if __name__ == "__main__":
  test.main()
