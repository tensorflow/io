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
"""Tests for TFRecordWriter"""

import os
import pytest
import shutil
import tempfile
import tensorflow as tf

from tensorflow_io.python.experimental.benchmark.data_source import \
  DataSource
from tensorflow_io.python.experimental.benchmark.generator.\
  tensor_generator import IntTensorGenerator, FloatTensorGenerator, \
    WordTensorGenerator, BoolTensorGenerator
from tensorflow_io.python.experimental.benchmark.generator. \
  sparse_tensor_generator import ValueDistribution, IntSparseTensorGenerator, \
  FloatSparseTensorGenerator, WordSparseTensorGenerator, \
  BoolSparseTensorGenerator
from tensorflow_io.python.experimental.benchmark.generator. \
  varlen_tensor_generator import IntVarLenTensorGenerator, \
  FloatVarLenTensorGenerator, WordVarLenTensorGenerator, \
  BoolVarLenTensorGenerator
from tensorflow_io.python.experimental.benchmark.tf_record_writer import \
  TFRecordWriter
from .generator.mock_generator import MockGenerator


@pytest.mark.parametrize(
  ["num_records", "partitions"], [
    (10, 1),
    (23, 3),
    (5, 6)
  ]
)
def test_expected_num_records_and_partitions(num_records, partitions):
  feature_name = "feature"
  scenario = {
    feature_name: IntTensorGenerator(tf.TensorSpec(shape=[3], dtype=tf.int64))
  }
  data_source = DataSource(
    scenario=scenario,
    num_records=num_records,
    partitions=partitions
  )

  with TFRecordWriter() as writer:
    dir_path = writer.write(data_source)
    pattern = os.path.join(dir_path, f"*.{writer.extension}")

    dataset = tf.data.Dataset.list_files(pattern)
    assert dataset.cardinality().numpy() == partitions

    parse_function = writer.create_tf_example_parser_fn(data_source)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(parse_function)
    counts = 0
    for _ in dataset:
      counts = counts + 1
    assert counts == num_records


@pytest.mark.parametrize(
  ["generator"], [
    (IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int32)), ),
    (IntTensorGenerator(tf.TensorSpec(shape=[3], dtype=tf.int32)), ),
    (IntTensorGenerator(tf.TensorSpec(shape=[3, 8], dtype=tf.int32)), ),
    (IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int64)), ),
    (IntTensorGenerator(tf.TensorSpec(shape=[5], dtype=tf.int64)), ),
    (IntTensorGenerator(tf.TensorSpec(shape=[1, 2], dtype=tf.int64)), ),
    (FloatTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.float32)), ),
    (FloatTensorGenerator(tf.TensorSpec(shape=[10], dtype=tf.float32)), ),
    (FloatTensorGenerator(tf.TensorSpec(shape=[2, 4, 6], dtype=tf.float32)), ),
    (FloatTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.float64)), ),
    (FloatTensorGenerator(tf.TensorSpec(shape=[1], dtype=tf.float64)), ),
    (FloatTensorGenerator(tf.TensorSpec(shape=[2, 4], dtype=tf.float64)), ),
    (WordTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.string)), ),
    (WordTensorGenerator(tf.TensorSpec(shape=[5], dtype=tf.string)), ),
    (WordTensorGenerator(tf.TensorSpec(shape=[2, 1], dtype=tf.string)), ),
    (BoolTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.bool)), ),
    (BoolTensorGenerator(tf.TensorSpec(shape=[5], dtype=tf.bool)), ),
    (BoolTensorGenerator(tf.TensorSpec(shape=[2, 1, 3], dtype=tf.bool)), ),
  ]
)
def test_dense_tensor_with_various_spec(generator):
  feature_name = "feature"
  num_records = 10
  data = [generator.generate() for _ in range(num_records)]

  mock_generator = MockGenerator(spec=generator.spec, data=data, generator_cls=type(generator))
  data_source = DataSource(
    scenario={feature_name: mock_generator},
    num_records=num_records,
  )

  dtype = generator.spec.dtype
  with TFRecordWriter() as writer:
    dir_path = writer.write(data_source)
    pattern = os.path.join(dir_path, f"*.{writer.extension}")

    parse_function = writer.create_tf_example_parser_fn(data_source)
    dataset = tf.data.Dataset.list_files(pattern)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(parse_function)
    for i, features in enumerate(dataset):
      # tf.Example only supports tf.float32, tf.int64, and tf.string
      # For other dtypes, cast feature into its original dtype.
      actual = tf.cast(features[feature_name], dtype)
      if dtype in [tf.float32, tf.float64]:
        tf.debugging.assert_near(actual, data[i], atol=1e-6)
      else:
        tf.debugging.assert_equal(actual, data[i])


@pytest.mark.parametrize(
  ["generator"], [
    (IntSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[10], dtype=tf.int32),
     num_values=ValueDistribution.SMALL_NUM_VALUE), ),
    (IntSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[2, 5], dtype=tf.int32),
     num_values=ValueDistribution.SINGLE_VALUE), ),
    (IntSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[100], dtype=tf.int64),
     num_values=ValueDistribution.SMALL_NUM_VALUE), ),
    (IntSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[20, 500], dtype=tf.int64),
     num_values=ValueDistribution.LARGE_NUM_VALUE), ),
    (FloatSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[20], dtype=tf.float32),
     num_values=ValueDistribution.SMALL_NUM_VALUE), ),
    (FloatSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[1, 10], dtype=tf.float32),
     num_values=ValueDistribution.SMALL_NUM_VALUE), ),
    (FloatSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[50000], dtype=tf.float64),
     num_values=ValueDistribution.LARGE_NUM_VALUE), ),
    (FloatSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[2, 2], dtype=tf.float64),
     num_values=ValueDistribution.SMALL_NUM_VALUE), ),
    (WordSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[5], dtype=tf.string),
     num_values=ValueDistribution.SMALL_NUM_VALUE), ),
    (WordSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[10, 3], dtype=tf.string),
     num_values=ValueDistribution.SMALL_NUM_VALUE), ),
    (BoolSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[1], dtype=tf.bool),
     num_values=ValueDistribution.SINGLE_VALUE), ),
    (BoolSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[1, 1], dtype=tf.bool),
     num_values=ValueDistribution.SINGLE_VALUE), ),
  ]
)
def test_sparse_tensor_with_various_spec(generator):
  feature_name = "feature"
  num_records = 10
  data = [generator.generate() for _ in range(num_records)]

  mock_generator = MockGenerator(spec=generator.spec, data=data, generator_cls=type(generator))
  data_source = DataSource(
    scenario={feature_name: mock_generator},
    num_records=num_records,
  )

  dtype = generator.spec.dtype
  with TFRecordWriter() as writer:
    dir_path = writer.write(data_source)
    pattern = os.path.join(dir_path, f"*.{writer.extension}")

    parse_function = writer.create_tf_example_parser_fn(data_source)
    dataset = tf.data.Dataset.list_files(pattern)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(parse_function)
    for i, features in enumerate(dataset):
      # tf.Example only supports tf.float32, tf.int64, and tf.string
      # For other dtypes, cast feature into its original dtype.
      sparse_tensor = tf.cast(features[feature_name], dtype)
      tf.debugging.assert_equal(sparse_tensor.indices, data[i].indices)
      tf.debugging.assert_equal(sparse_tensor.dense_shape, data[i].dense_shape)
      if dtype in [tf.float32, tf.float64]:
        tf.debugging.assert_near(
          sparse_tensor.values, data[i].values, atol=1e-6)
      else:
        tf.debugging.assert_equal(sparse_tensor.values, data[i].values)

@pytest.mark.parametrize(
  ["generator"], [
    (IntVarLenTensorGenerator(tf.SparseTensorSpec(shape=[10], dtype=tf.int32)), ),
    (IntVarLenTensorGenerator(tf.SparseTensorSpec(shape=[None], dtype=tf.int32)), ),
    (IntVarLenTensorGenerator(tf.SparseTensorSpec(shape=[100], dtype=tf.int64)), ),
    (IntVarLenTensorGenerator(tf.SparseTensorSpec(shape=[None], dtype=tf.int64)), ),
    (FloatVarLenTensorGenerator(tf.SparseTensorSpec(shape=[None], dtype=tf.float32)), ),
    (FloatVarLenTensorGenerator(tf.SparseTensorSpec(shape=[50], dtype=tf.float32)), ),
    (FloatVarLenTensorGenerator(tf.SparseTensorSpec(shape=[50000], dtype=tf.float64)), ),
    (FloatVarLenTensorGenerator(tf.SparseTensorSpec(shape=[None], dtype=tf.float64)), ),
    (WordVarLenTensorGenerator(tf.SparseTensorSpec(shape=[5], dtype=tf.string)), ),
    (WordVarLenTensorGenerator(tf.SparseTensorSpec(shape=[None], dtype=tf.string)), ),
    (BoolVarLenTensorGenerator(tf.SparseTensorSpec(shape=[None], dtype=tf.bool)), ),
    (BoolVarLenTensorGenerator(tf.SparseTensorSpec(shape=[1], dtype=tf.bool)), ),
  ]
)
def test_varlen_tensor_with_various_spec(generator):
  feature_name = "feature"
  num_records = 10
  data = [generator.generate() for _ in range(num_records)]
  mock_generator = MockGenerator(spec=generator.spec, data=data, generator_cls=type(generator))
  data_source = DataSource(
    scenario={feature_name: mock_generator},
    num_records=num_records,
  )

  dtype = generator.spec.dtype
  with TFRecordWriter() as writer:
    dir_path = writer.write(data_source)
    pattern = os.path.join(dir_path, f"*.{writer.extension}")

    parse_function = writer.create_tf_example_parser_fn(data_source)
    dataset = tf.data.Dataset.list_files(pattern)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(parse_function)
    for i, features in enumerate(dataset):
      # tf.Example only supports tf.float32, tf.int64, and tf.string
      # For other dtypes, cast feature into its original dtype.
      sparse_tensor = tf.cast(features[feature_name], dtype)
      generator.spec.is_compatible_with(sparse_tensor.values)

@pytest.mark.parametrize(
  ["generator"], [
    (IntSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[None], dtype=tf.int32),
     num_values=ValueDistribution.SINGLE_VALUE), ),
    (IntSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[100, None], dtype=tf.int64),
     num_values=ValueDistribution.SMALL_NUM_VALUE), ),
    (FloatSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[None, 4], dtype=tf.float32),
     num_values=ValueDistribution.SINGLE_VALUE), ),
    (FloatSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[None, 10], dtype=tf.float64),
     num_values=ValueDistribution.SMALL_NUM_VALUE), ),
    (WordSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[None], dtype=tf.string),
     num_values=ValueDistribution.SINGLE_VALUE), ),
    (BoolSparseTensorGenerator(
     tf.SparseTensorSpec(shape=[None, 4], dtype=tf.bool),
     num_values=ValueDistribution.SMALL_NUM_VALUE), ),
    (IntVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[None, None], dtype=tf.int32)), ),
    (IntVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[None, 10], dtype=tf.int32)), ),
    (IntVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[100, 10], dtype=tf.int64)), ),
    (IntVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[20, None], dtype=tf.int64)), ),
    (FloatVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[100, None], dtype=tf.float32)), ),
    (FloatVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32)), ),
    (FloatVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[100, None], dtype=tf.float64)), ),
    (FloatVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[None, 20, None], dtype=tf.float64)), ),
    (WordVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[5, 5], dtype=tf.string)), ),
    (WordVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[None, 3], dtype=tf.string)), ),
    (BoolVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[10, None], dtype=tf.bool)), ),
    (BoolVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[None, 1], dtype=tf.bool)), ),
  ]
)
def test_unsupported_spec(generator):
  feature_name = "feature"
  num_records = 10
  data = [generator.generate() for _ in range(num_records)]

  mock_generator = MockGenerator(spec=generator.spec, data=data, generator_cls=type(generator))
  data_source = DataSource(
    scenario={feature_name: mock_generator},
    num_records=num_records,
  )

  with TFRecordWriter() as writer:
    error_message = "SparseTensorSpec shape must be either a 1D varlen tensor from VarLenTensorGenerator " \
                    "or fully defined sparse tensor from SparseTensorGenerator. Found *"
    with pytest.raises(ValueError, match=error_message):
      writer.write(data_source)

def test_hash_code_equal():
  writer_1 = TFRecordWriter()
  writer_2 = TFRecordWriter()
  assert writer_1.hash_code() == writer_2.hash_code()

@pytest.mark.parametrize(
  ["generator", "num_records", "partitions"], [
    (IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int32)), 10, 3),
    (IntTensorGenerator(tf.TensorSpec(shape=[3], dtype=tf.int32)), 5, 1),
    (IntTensorGenerator(tf.TensorSpec(shape=[3, 8], dtype=tf.int32)), 2, 1),
    (IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int64)), 10, 4),
    (IntTensorGenerator(tf.TensorSpec(shape=[5], dtype=tf.int64)), 20, 3),
    (IntTensorGenerator(tf.TensorSpec(shape=[1, 2], dtype=tf.int64)), 20, 1),
    (FloatTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.float32)), 10, 1),
    (FloatTensorGenerator(tf.TensorSpec(shape=[10], dtype=tf.float32)), 50, 4),
    (FloatTensorGenerator(tf.TensorSpec(shape=[2, 4, 6], dtype=tf.float32)), 40, 4),
    (FloatTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.float64)), 10, 1),
    (FloatTensorGenerator(tf.TensorSpec(shape=[1], dtype=tf.float64)), 10, 4),
    (FloatTensorGenerator(tf.TensorSpec(shape=[2, 4], dtype=tf.float64)), 5, 5),
    (WordTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.string)), 4, 1),
    (WordTensorGenerator(tf.TensorSpec(shape=[5], dtype=tf.string)), 5, 1),
    (WordTensorGenerator(tf.TensorSpec(shape=[2, 1], dtype=tf.string)), 1, 1),
    (BoolTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.bool)), 2, 1),
    (BoolTensorGenerator(tf.TensorSpec(shape=[5], dtype=tf.bool)), 10, 2),
    (BoolTensorGenerator(tf.TensorSpec(shape=[2, 1, 3], dtype=tf.bool)), 3, 2),
  ]
)
def test_write_to_path_from_cached_data(generator, num_records, partitions):
  feature_name = "feature"
  data = [generator.generate() for _ in range(num_records)]
  mock_generator = MockGenerator(spec=generator.spec, data=data, generator_cls=type(generator))
  data_source = DataSource(
    scenario={feature_name: mock_generator},
    num_records=num_records,
    partitions=partitions
  )

  dtype = generator.spec.dtype
  try:
    with TFRecordWriter() as writer:
      dir_path = writer.write(data_source)
      pattern = os.path.join(dir_path, f"*.{writer.extension}")

      parse_function = writer.create_tf_example_parser_fn(data_source)
      dataset = tf.data.Dataset.list_files(pattern, shuffle=False)
      dataset = tf.data.TFRecordDataset(dataset)
      dataset = dataset.map(parse_function)

      cached_dir_path = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))
      writer._write_to_path_from_cached_data(cached_dir_path, data_source, dataset)
      cached_pattern = os.path.join(cached_dir_path, f"*.{writer.extension}")
      cached_dataset = tf.data.Dataset.list_files(cached_pattern, shuffle=False)
      cached_dataset = tf.data.TFRecordDataset(cached_dataset)
      cached_dataset = cached_dataset.map(parse_function)
      for i, cached_dataset_record in enumerate(cached_dataset):
        # tf.Example only supports tf.float32, tf.int64, and tf.string
        # For other dtypes, cast feature into its original dtype.
        cached_actual = tf.cast(cached_dataset_record[feature_name], dtype)
        if dtype in [tf.float32, tf.float64]:
          tf.debugging.assert_near(cached_actual, data[i], atol=1e-6)
        else:
          tf.debugging.assert_equal(cached_actual, data[i])
  finally:
    shutil.rmtree(cached_dir_path)
