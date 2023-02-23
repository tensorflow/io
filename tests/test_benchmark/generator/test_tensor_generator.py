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
"""Tests for various TensorGenerator"""

import re
import pytest
import tensorflow as tf
from tensorflow.python.framework import errors

from tensorflow_io.python.experimental.benchmark.generator.\
  tensor_generator import IntTensorGenerator, FloatTensorGenerator, \
    WordTensorGenerator, BoolTensorGenerator


@pytest.mark.parametrize(
  ["cls", "spec"], [
    (IntTensorGenerator, tf.TensorSpec(shape=[], dtype=tf.int32)),
    (IntTensorGenerator, tf.TensorSpec(shape=[10], dtype=tf.int32)),
    (IntTensorGenerator, tf.TensorSpec(shape=[4, 16], dtype=tf.int32)),
    (IntTensorGenerator, tf.TensorSpec(shape=[], dtype=tf.int64)),
    (IntTensorGenerator, tf.TensorSpec(shape=[100], dtype=tf.int64)),
    (IntTensorGenerator, tf.TensorSpec(shape=[2, 5, 2], dtype=tf.int64)),
    (FloatTensorGenerator, tf.TensorSpec(shape=[], dtype=tf.float32)),
    (FloatTensorGenerator, tf.TensorSpec(shape=[1], dtype=tf.float32)),
    (FloatTensorGenerator, tf.TensorSpec(shape=[2, 5], dtype=tf.float32)),
    (FloatTensorGenerator, tf.TensorSpec(shape=[], dtype=tf.float64)),
    (FloatTensorGenerator, tf.TensorSpec(shape=[7], dtype=tf.float64)),
    (FloatTensorGenerator, tf.TensorSpec(shape=[5, 10], dtype=tf.float64)),
    (WordTensorGenerator, tf.TensorSpec(shape=[], dtype=tf.string)),
    (WordTensorGenerator, tf.TensorSpec(shape=[5], dtype=tf.string)),
    (WordTensorGenerator, tf.TensorSpec(shape=[3, 4], dtype=tf.string)),
    (BoolTensorGenerator, tf.TensorSpec(shape=[], dtype=tf.bool)),
    (BoolTensorGenerator, tf.TensorSpec(shape=[1], dtype=tf.bool)),
    (BoolTensorGenerator, tf.TensorSpec(shape=[1, 1], dtype=tf.bool)),
  ]
)
def test_tensor_generator_compatible(cls, spec):
  generator = cls(spec)
  data = generator.generate()
  assert spec.is_compatible_with(data)


@pytest.mark.parametrize(
  ["cls", "spec"], [
    (IntTensorGenerator, tf.SparseTensorSpec(shape=[], dtype=tf.int32)),
    (FloatTensorGenerator, tf.SparseTensorSpec(shape=[], dtype=tf.float32)),
    (WordTensorGenerator, tf.RaggedTensorSpec(shape=[], dtype=tf.string)),
    (BoolTensorGenerator, tf.RaggedTensorSpec(shape=[], dtype=tf.bool)),
  ]
)
def test_input_not_tensor_spec(cls, spec):
  error_message = "Input spec must be a tf.TensorSpec in TensorGenerator *"
  with pytest.raises(TypeError, match=error_message):
    generator = cls(spec)


@pytest.mark.parametrize(
  ["cls", "spec"], [
    (IntTensorGenerator, tf.TensorSpec(shape=[None], dtype=tf.int32)),
    (FloatTensorGenerator, tf.TensorSpec(shape=[1, None], dtype=tf.float32)),
    (WordTensorGenerator, tf.TensorSpec(shape=[None, 2], dtype=tf.string)),
    (BoolTensorGenerator, tf.TensorSpec(shape=[None, None], dtype=tf.bool)),
  ]
)
def test_shape_not_fully_defined(cls, spec):
  error_message = r"Shape .* is not fully defined"
  with pytest.raises(ValueError, match=error_message):
    generator = cls(spec)


@pytest.mark.parametrize(
  ["cls", "spec", "error_message"], [
    (IntTensorGenerator, tf.TensorSpec(shape=[], dtype=tf.float32),
     "IntTensorGenerator can only generate tf.Tensor with dtype in tf.int32 "
     "or tf.int64 *"),
    (FloatTensorGenerator, tf.TensorSpec(shape=[], dtype=tf.int32),
     "FloatTensorGenerator can only generate tf.Tensor with dtype in "
     "tf.float32 or tf.float64 *"),
    (WordTensorGenerator, tf.TensorSpec(shape=[], dtype=tf.bool),
     "WordTensorGenerator can only generate tf.Tensor with dtype in "
     "tf.string *"),
    (BoolTensorGenerator, tf.TensorSpec(shape=[], dtype=tf.string),
     "BoolTensorGenerator can only generate tf.Tensor with dtype in "
     "tf.bool *"),
  ]
)
def test_invalid_dtype(cls, spec, error_message):
  with pytest.raises(TypeError, match=error_message):
    generator = cls(spec)


def test_invalid_avg_length():
  spec = tf.TensorSpec(shape=[], dtype=tf.string)
  with pytest.raises(ValueError, match=""):
    WordTensorGenerator(spec, avg_length=0)
