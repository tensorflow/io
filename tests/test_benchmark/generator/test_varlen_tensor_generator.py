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
"""Tests for various VarLenTensorGenerator"""

import re
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import errors

from tensorflow_io.python.experimental.benchmark.generator import sparse_util
from tensorflow_io.python.experimental.benchmark.generator.\
  varlen_tensor_generator import IntVarLenTensorGenerator, FloatVarLenTensorGenerator, \
    WordVarLenTensorGenerator, BoolVarLenTensorGenerator


@pytest.mark.parametrize(
  ["cls", "spec"], [
    (IntVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.int32)),
    (IntVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None, None], dtype=tf.int64)),
    (FloatVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.float32)),
    (FloatVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None, None, None], dtype=tf.float64)),
    (WordVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None, 5], dtype=tf.string)),
    (WordVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.string)),
    (BoolVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.bool)),
    (BoolVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None, None], dtype=tf.bool)),
  ]
)
def test_get_idx(cls, spec):
  generator = cls(spec)
  idx = []
  shape = generator._get_shape()
  rank = len(shape)
  generator._get_idx(0, shape, [0] * rank, idx)
  cur_dim = [0] * (rank-1)
  cur_idx = 0
  # Check that indices are generated in row-major order, and that the last dimension
  # is in increasing contiguous order starting from 0, e.g.
  # idx = [[0, 1, 3, 5, 0],
  #        [0, 1, 3, 5, 1],
  #        [0, 1, 4, 1, 0],
  #        [0, 1, 4, 1, 1],
  #        [0, 1, 4, 1, 2]]
  for i in idx:
    outer_dim = i[0:(rank-1)]
    if outer_dim != cur_dim:
      assert sparse_util.coord_to_int(cur_dim, shape[0:(rank-1)]) < sparse_util.coord_to_int(outer_dim, shape[0:(rank-1)])
      cur_dim = outer_dim
      cur_idx = 0
    assert i[rank-1] == cur_idx
    cur_idx += 1


@pytest.mark.parametrize(
  ["cls", "spec"], [
    (IntVarLenTensorGenerator, tf.SparseTensorSpec(shape=[10], dtype=tf.int32)),
    (IntVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.int32)),
    (IntVarLenTensorGenerator, tf.SparseTensorSpec(shape=[2, None], dtype=tf.int64)),
    (IntVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.int64)),
    (FloatVarLenTensorGenerator, tf.SparseTensorSpec(shape=[1], dtype=tf.float32)),
    (FloatVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None, None, None], dtype=tf.float64)),
    (WordVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None, 5], dtype=tf.string)),
    (WordVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.string)),
    (BoolVarLenTensorGenerator, tf.SparseTensorSpec(shape=[1], dtype=tf.bool)),
    (BoolVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None, None], dtype=tf.bool)),
  ]
)
def test_tensor_generator_compatible(cls, spec):
  generator = cls(spec)
  data = generator.generate()
  assert spec.is_compatible_with(data)

@pytest.mark.parametrize(
  ["cls", "spec"], [
    (IntVarLenTensorGenerator, tf.TensorSpec(shape=[2], dtype=tf.int32)),
    (FloatVarLenTensorGenerator, tf.TensorSpec(shape=[5], dtype=tf.float32)),
    (WordVarLenTensorGenerator, tf.RaggedTensorSpec(shape=[None], dtype=tf.string)),
    (BoolVarLenTensorGenerator, tf.RaggedTensorSpec(shape=[None], dtype=tf.bool)),
  ]
)
def test_wrong_tensor_spec_type(cls, spec):
  error_message = "Input spec must be a tf.SparseTensorSpec in VarLenTensorGenerator *"
  with pytest.raises(TypeError, match=error_message):
    generator = cls(spec)

@pytest.mark.parametrize(
  ["cls", "spec", "error_message"], [
    (IntVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.float32),
     "IntVarLenTensorGenerator can only generate tf.sparse.SparseTensor with dtype in tf.int32 "
     "or tf.int64 *"),
    (FloatVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.int32),
     "FloatVarLenTensorGenerator can only generate tf.sparse.SparseTensor with dtype in "
     "tf.float32 or tf.float64 *"),
    (WordVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.bool),
     "WordVarLenTensorGenerator can only generate tf.sparse.SparseTensor with dtype in "
     "tf.string *"),
    (BoolVarLenTensorGenerator, tf.SparseTensorSpec(shape=[None], dtype=tf.string),
     "BoolVarLenTensorGenerator can only generate tf.sparse.SparseTensor with dtype in "
     "tf.bool *"),
  ]
)
def test_invalid_dtype(cls, spec, error_message):
  with pytest.raises(TypeError, match=error_message):
    generator = cls(spec)
