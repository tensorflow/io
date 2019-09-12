# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for CSV"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np
import pandas as pd

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io as tfio # pylint: disable=wrong-import-position

def test_csv_format():
  """test_csv_format"""
  data = {
      'bool': np.asarray([e%2 for e in range(100)], np.bool),
      'int64': np.asarray(range(100), np.int64),
      'double': np.asarray(range(100), np.float64),
  }
  df = pd.DataFrame(data).sort_index(axis=1)
  with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
    df.to_csv(f, index=False)

  df = pd.read_csv(f.name)

  csv = tfio.IOTensor.from_csv(f.name)
  for column in df.columns:
    assert csv(column).shape == [100]
    assert csv(column).dtype == column
    assert np.all(csv(column).to_tensor().numpy() == data[column])

  os.unlink(f.name)

def test_null_csv_format():
  """test_null_csv_format"""
  # cat tests/test_csv/null.csv
  # C1,C2,C3
  # 1,2,3
  # 4,NaN,6
  # 7,8,9
  csv_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_csv", "null.csv")
  csv = tfio.IOTensor.from_csv(csv_path)
  assert np.all(csv.isnull('C2').to_tensor().numpy() == [False, True, False])

if __name__ == "__main__":
  test.main()
