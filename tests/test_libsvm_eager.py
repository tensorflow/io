# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for DecodeLibsvm op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio

def test_basic():
  """test_basic"""
  content = [
      "1 1:3.4 2:0.5 4:0.231", "1 2:2.5 3:inf 5:0.503",
      "2 3:2.5 2:nan 1:0.105"
  ]
  sparse_features, labels = tfio.experimental.text.decode_libsvm(
      content, num_features=6)
  features = tf.sparse.to_dense(
      sparse_features, validate_indices=False)

  assert labels.shape == [3]

  assert np.all(labels == [1, 1, 2])

  assert features.shape == [3, 6]
  expected = [[0, 3.4, 0.5, 0, 0.231, 0],
              [0, 0, 2.5, np.inf, 0, 0.503],
              [0, 0.105, np.nan, 2.5, 0, 0]]
  for i, e in enumerate(expected):
    for j, v in enumerate(e):
      assert features[i, j] == v or (np.isnan(features[i, j]) and np.isnan(v))

def test_n_dimension():
  """test_n_dimension"""
  content = [["1 1:3.4 2:0.5 4:0.231", "1 1:3.4 2:0.5 4:0.231"],
             ["1 2:2.5 3:inf 5:0.503", "1 2:2.5 3:inf 5:0.503"],
             ["2 3:2.5 2:nan 1:0.105", "2 3:2.5 2:nan 1:0.105"]]
  sparse_features, labels = tfio.experimental.text.decode_libsvm(
      content, num_features=6, label_dtype=tf.float64)
  features = tf.sparse.to_dense(
      sparse_features, validate_indices=False)

  assert labels.shape == [3, 2]

  np.all(labels == [[1, 1], [1, 1], [2, 2]])

  expected = [[[0, 3.4, 0.5, 0, 0.231, 0], [0, 3.4, 0.5, 0, 0.231, 0]],
              [[0, 0, 2.5, np.inf, 0, 0.503], [0, 0, 2.5, np.inf, 0, 0.503]],
              [[0, 0.105, np.nan, 2.5, 0, 0], [0, 0.105, np.nan, 2.5, 0, 0]]]
  np.all(features == expected)

def test_dataset():
  """test_dataset"""
  libsvm_file = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_libsvm", "sample")
  dataset = tfio.experimental.IODataset.from_libsvm(
      libsvm_file, num_features=6)
  dataset = dataset.map(
      lambda sparse_features, labels: (
          tf.sparse.to_dense(sparse_features, validate_indices=False), labels))
  expected_f = [
      [0., 3.4, 0.5, 0., 0.231, 0.],
      [0, 0, 2.5, np.inf, 0, 0.503],
      [0, 0.105, np.nan, 2.5, 0, 0]]
  expected_l = [[1], [1], [2]]
  i = 0
  for f, l in dataset:
    for j, v in enumerate(f[0]):
      assert np.isclose(expected_f[i][j], v.numpy()) or (
          np.isnan(expected_f[i][j]) and np.isnan(v.numpy()))
    assert np.allclose(expected_l[i], l)
    i += 1

if __name__ == "__main__":
  test.main()
