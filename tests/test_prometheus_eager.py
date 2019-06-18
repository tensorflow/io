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

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io.prometheus as prometheus_io # pylint: disable=wrong-import-position

if sys.platform == "darwin":
  pytest.skip(
      "prometheus is not supported on macOS yet", allow_module_level=True)

def test_prometheus_input():
  """test_prometheus_input
  """
  for _ in range(6):
    subprocess.call(["dig", "@localhost", "-p", "1053", "www.google.com"])
    time.sleep(1)
  time.sleep(2)
  prometheus_dataset = prometheus_io.PrometheusDataset(
      "http://localhost:9090",
      schema="coredns_dns_request_count_total[5s]",
      batch=2)
  i = 0
  for k, v in prometheus_dataset:
    print("K, V: ", k.numpy(), v.numpy())
    if i == 4:
      # Last entry guaranteed 6.0
      assert v.numpy() == 6.0
    i += 2
  assert i == 6

if __name__ == "__main__":
  test.main()
