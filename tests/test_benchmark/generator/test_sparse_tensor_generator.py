# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for various SparseTensorGenerator"""

import re
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import errors

from tensorflow_io.python.experimental.benchmark.generator import sparse_util
from tensorflow_io.python.experimental.benchmark.generator.\
  sparse_tensor_generator import IntSparseTensorGenerator, FloatSparseTensorGenerator, ValueDistribution, \
    WordSparseTensorGenerator, BoolSparseTensorGenerator


@pytest.mark.parametrize(
  ["cls", "spec", "expected_idx", "expected_coord"], [
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[3, 5], dtype=tf.int32), 7, [1, 2]),
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[2, 3, 4], dtype=tf.int64), 17, [1, 1, 1]),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[2, 5], dtype=tf.float32), 3, [0, 3]),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[7], dtype=tf.float64), 5, [5]),
    (WordSparseTensorGenerator, tf.SparseTensorSpec(shape=[1, 3, 5], dtype=tf.string), 13, [0, 2, 3]),
    (BoolSparseTensorGenerator, tf.SparseTensorSpec(shape=[2, 4, 6], dtype=tf.bool), 43, [1, 3, 1]),
  ]
)
def test_idx_coord_bijection(cls, spec, expected_idx, expected_coord):
  generator = cls(spec, 1)
  coord = generator._int_to_coord(expected_idx, spec.shape)
  assert coord == expected_coord
  idx = sparse_util.coord_to_int(expected_coord, spec.shape)
  assert idx == expected_idx


@pytest.mark.parametrize(
  ["cls", "spec", "num_values"], [
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[10], dtype=tf.int32), 2),
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[4, 16], dtype=tf.int32), 5),
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[100], dtype=tf.int64), 7),
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[2, 5, 2], dtype=tf.int64), 9),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[1], dtype=tf.float32), 1),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[2, 5], dtype=tf.float32), 3),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[7], dtype=tf.float64), 5),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[5, 10, 6], dtype=tf.float64), 20),
    (WordSparseTensorGenerator, tf.SparseTensorSpec(shape=[5], dtype=tf.string), 2),
    (WordSparseTensorGenerator, tf.SparseTensorSpec(shape=[3, 4], dtype=tf.string), 3),
    (BoolSparseTensorGenerator, tf.SparseTensorSpec(shape=[1], dtype=tf.bool), 0),
    (BoolSparseTensorGenerator, tf.SparseTensorSpec(shape=[1, 1], dtype=tf.bool), 1),
  ]
)
def test_tensor_generator_compatible(cls, spec, num_values):
  generator = cls(spec, num_values)
  data = generator.generate()
  assert spec.is_compatible_with(data)
  coords = data.indices.numpy()
  indices = [sparse_util.coord_to_int(coord, spec.shape) for coord in coords]
  # Indices must be unique
  assert len(indices) == len(set(indices))
  # Indices should be sorted in row-major order by sparse tensor convention
  assert indices == sorted(indices)

@pytest.mark.parametrize(
  ["cls", "spec"], [
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[10], dtype=tf.int32)),
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[2, 5, 2], dtype=tf.int64)),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[20], dtype=tf.float32)),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[5, 10, 6], dtype=tf.float64)),
    (WordSparseTensorGenerator, tf.SparseTensorSpec(shape=[15], dtype=tf.string)),
    (BoolSparseTensorGenerator, tf.SparseTensorSpec(shape=[20, 10], dtype=tf.bool)),
  ]
)
def test_single_value_distribution(cls, spec):
  generator = cls(spec, ValueDistribution.SINGLE_VALUE)
  data = generator.generate()
  assert spec.is_compatible_with(data)
  num_values = len(data.values)
  num_indices = len(data.indices)
  assert num_values == 1
  assert num_values == num_indices

@pytest.mark.parametrize(
  ["cls", "spec"], [
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[10], dtype=tf.int32)),
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[2, 5, 2], dtype=tf.int64)),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[20], dtype=tf.float32)),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[5, 10, 6], dtype=tf.float64)),
    (WordSparseTensorGenerator, tf.SparseTensorSpec(shape=[15], dtype=tf.string)),
    (BoolSparseTensorGenerator, tf.SparseTensorSpec(shape=[20, 10], dtype=tf.bool)),
  ]
)
def test_small_value_distribution(cls, spec):
  generator = cls(spec, ValueDistribution.SMALL_NUM_VALUE)
  data = generator.generate()
  assert spec.is_compatible_with(data)
  num_values = len(data.values)
  num_indices = len(data.indices)
  assert num_values >= 5 and num_values < 10
  assert num_values == num_indices

@pytest.mark.parametrize(
  ["cls", "spec"], [
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[1001], dtype=tf.int32)),
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[10, 10, 11], dtype=tf.int64)),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[10000], dtype=tf.float32)),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[10, 101], dtype=tf.float64)),
    (WordSparseTensorGenerator, tf.SparseTensorSpec(shape=[100000], dtype=tf.string)),
    (BoolSparseTensorGenerator, tf.SparseTensorSpec(shape=[50, 100], dtype=tf.bool)),
  ]
)
def test_large_value_distribution(cls, spec):
  generator = cls(spec, ValueDistribution.LARGE_NUM_VALUE)
  data = generator.generate()
  assert spec.is_compatible_with(data)
  num_values = len(data.values)
  num_indices = len(data.indices)
  assert num_values >= 100 and num_values < 1000
  assert num_values == num_indices

@pytest.mark.parametrize(
  ["cls", "spec"], [
    (IntSparseTensorGenerator, tf.TensorSpec(shape=[2], dtype=tf.int32)),
    (FloatSparseTensorGenerator, tf.TensorSpec(shape=[5], dtype=tf.float32)),
    (WordSparseTensorGenerator, tf.RaggedTensorSpec(shape=[1, 2], dtype=tf.string)),
    (BoolSparseTensorGenerator, tf.RaggedTensorSpec(shape=[3, 4], dtype=tf.bool)),
  ]
)
def test_wrong_tensor_spec_type(cls, spec):
  error_message = "Input spec must be a tf.SparseTensorSpec in SparseTensorGenerator *"
  with pytest.raises(TypeError, match=error_message):
    generator = cls(spec, 1)

@pytest.mark.parametrize(
  ["cls", "spec", "error_message"], [
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[3], dtype=tf.float32),
     "IntSparseTensorGenerator can only generate tf.sparse.SparseTensor with dtype in tf.int32 "
     "or tf.int64 *"),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[4, 5], dtype=tf.int32),
     "FloatSparseTensorGenerator can only generate tf.sparse.SparseTensor with dtype in "
     "tf.float32 or tf.float64 *"),
    (WordSparseTensorGenerator, tf.SparseTensorSpec(shape=[1], dtype=tf.bool),
     "WordSparseTensorGenerator can only generate tf.sparse.SparseTensor with dtype in "
     "tf.string *"),
    (BoolSparseTensorGenerator, tf.SparseTensorSpec(shape=[2], dtype=tf.string),
     "BoolSparseTensorGenerator can only generate tf.sparse.SparseTensor with dtype in "
     "tf.bool *"),
  ]
)
def test_invalid_dtype(cls, spec, error_message):
  with pytest.raises(TypeError, match=error_message):
    generator = cls(spec, 1)

@pytest.mark.parametrize(
  ["cls", "spec", "num_values"], [
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[3], dtype=tf.int32), 4),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[4, 5], dtype=tf.float32), 21),
    (WordSparseTensorGenerator, tf.SparseTensorSpec(shape=[2, 3, 4], dtype=tf.string), 26),
    (BoolSparseTensorGenerator, tf.SparseTensorSpec(shape=[2], dtype=tf.bool), 4),
  ]
)
def test_too_many_values(cls, spec, num_values):
  generator = cls(spec, num_values)
  data = generator.generate()
  assert len(data.values) == np.prod(spec.shape.as_list())

@pytest.mark.parametrize(
  ["cls", "spec"], [
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.int32)),
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[4, None], dtype=tf.int32)),
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.int64)),
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=[None, 5, 2], dtype=tf.int64)),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.float32)),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[2, None], dtype=tf.float32)),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.float64)),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=[None, None, 6], dtype=tf.float64)),
    (WordSparseTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.string)),
    (WordSparseTensorGenerator, tf.SparseTensorSpec(shape=[3, None], dtype=tf.string)),
    (BoolSparseTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.bool)),
    (BoolSparseTensorGenerator, tf.SparseTensorSpec(shape=[1, None], dtype=tf.bool)),
  ]
)
def test_unknown_dimension(cls, spec):
  generator = cls(spec, 1)
  data = generator.generate()
  assert spec.is_compatible_with(data)
  shape = data.dense_shape.numpy()
  assert all(dim >= 1 and dim < 10 for dim in shape)
  coords = data.indices.numpy()
  indices = [sparse_util.coord_to_int(coord, shape) for coord in coords]
  # Indices must be unique
  assert len(indices) == len(set(indices))
  # Indices should be sorted in row-major order by sparse tensor convention
  assert indices == sorted(indices)

@pytest.mark.parametrize(
  ["cls", "spec"], [
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=None, dtype=tf.int32)),
    (IntSparseTensorGenerator, tf.SparseTensorSpec(shape=None, dtype=tf.int64)),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=None, dtype=tf.float32)),
    (FloatSparseTensorGenerator, tf.SparseTensorSpec(shape=None, dtype=tf.float64)),
    (WordSparseTensorGenerator, tf.SparseTensorSpec(shape=None, dtype=tf.string)),
    (BoolSparseTensorGenerator, tf.SparseTensorSpec(shape=None, dtype=tf.bool)),
  ]
)
def test_unknown_shape(cls, spec):
  generator = cls(spec, 1)
  data = generator.generate()
  assert spec.is_compatible_with(data)
  shape = data.dense_shape.numpy()
  assert len(shape) >= 1 and len(shape) < 5
  assert all(dim >= 1 and dim < 10 for dim in shape)
  coords = data.indices.numpy()
  indices = [sparse_util.coord_to_int(coord, shape) for coord in coords]
  # Indices must be unique
  assert len(indices) == len(set(indices))
  # Indices should be sorted in row-major order by sparse tensor convention
  assert indices == sorted(indices)
