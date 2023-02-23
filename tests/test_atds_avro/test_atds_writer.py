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
"""Tests for ATDSWriter"""

import os
import glob
import pytest
import shutil
import tempfile
import tensorflow as tf

from tensorflow_io.python.experimental.benchmark import file_writer
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
from tensorflow_io.python.experimental.atds.atds_writer import \
  ATDSWriter
from tensorflow_io.python.experimental.benchmark.tf_record_writer import \
  TFRecordWriter
from tests.test_atds_avro.utils.atds_benchmark_utils import \
  get_dataset, get_features_from_data_source
from tests.test_benchmark.generator.mock_generator import MockGenerator
from tests.test_parse_avro_eager import AvroFileToRecords


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

  with ATDSWriter() as writer:
    dir_path = writer.write(data_source)
    pattern = os.path.join(dir_path, f"*.{writer.extension}")

    dataset = tf.data.Dataset.list_files(pattern)
    assert dataset.cardinality().numpy() == partitions
    files = glob.glob(pattern)
    schema = writer.scenario_to_avro_schema(data_source.scenario)
    counts = 0
    for fname in files:
      assert os.path.isfile(fname), f"file does not exist: {fname}"
      assert os.stat(fname).st_size > 0, f"file is empty: {fname}"
      counts += len(AvroFileToRecords(fname, reader_schema=schema).get_records())
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
  with ATDSWriter() as writer:
    dir_path = writer.write(data_source)
    pattern = os.path.join(dir_path, f"*.{writer.extension}")
    dataset = get_dataset(glob.glob(pattern), get_features_from_data_source(writer, data_source))
    for i, features in enumerate(dataset):
      actual = features[feature_name]
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
  with ATDSWriter() as writer:
    dir_path = writer.write(data_source)
    pattern = os.path.join(dir_path, f"*.{writer.extension}")
    dataset = get_dataset(glob.glob(pattern), get_features_from_data_source(writer, data_source))
    dataset = dataset.unbatch()
    for i, features in enumerate(dataset):
      sparse_tensor = features[feature_name]
      tf.debugging.assert_equal(sparse_tensor.indices, data[i].indices)
      tf.debugging.assert_equal(sparse_tensor.dense_shape, data[i].dense_shape)

      if dtype in [tf.float32, tf.float64]:
        tf.debugging.assert_near(
          sparse_tensor.values, data[i].values, atol=1e-6)
      else:
        tf.debugging.assert_equal(sparse_tensor.values, data[i].values)

@pytest.mark.parametrize(
  ["generator"], [
    (IntVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[None], dtype=tf.int32)), ),
    (IntVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[None], dtype=tf.int64)), ),
    (FloatVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[20, None], dtype=tf.float32)), ),
    (FloatVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[1, 2], dtype=tf.float64)), ),
    (WordVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[None, None, None], dtype=tf.string)), ),
    (BoolVarLenTensorGenerator(
     tf.SparseTensorSpec(shape=[None, 1], dtype=tf.bool)), ),
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
  with ATDSWriter() as writer:
    dir_path = writer.write(data_source)
    pattern = os.path.join(dir_path, f"*.{writer.extension}")
    dataset = get_dataset(glob.glob(pattern), get_features_from_data_source(writer, data_source))
    dataset = dataset.unbatch()
    for i, features in enumerate(dataset):
      sparse_tensor = features[feature_name]
      tf.debugging.assert_equal(sparse_tensor.indices, data[i].indices)
      tf.debugging.assert_equal(sparse_tensor.dense_shape, data[i].dense_shape)

      if dtype in [tf.float32, tf.float64]:
        tf.debugging.assert_near(
          sparse_tensor.values, data[i].values, atol=1e-6)
      else:
        tf.debugging.assert_equal(sparse_tensor.values, data[i].values)


def test_hash_code():
  writer = ATDSWriter(codec="null")
  null_codec_writer = ATDSWriter(codec="null")
  deflate_codec_writer = ATDSWriter(codec="deflate")
  snappy_codec_writer = ATDSWriter(codec="snappy")

  assert writer.hash_code() == null_codec_writer.hash_code()
  assert writer.hash_code() != deflate_codec_writer.hash_code()
  assert writer.hash_code() != snappy_codec_writer.hash_code()
  assert deflate_codec_writer.hash_code() != snappy_codec_writer.hash_code()

@pytest.mark.parametrize(
  ["generator", "num_records", "partitions"], [
    (IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int32)), 10, 3),
    (IntTensorGenerator(tf.TensorSpec(shape=[3], dtype=tf.int32)), 10, 1),
    (IntTensorGenerator(tf.TensorSpec(shape=[3, 8], dtype=tf.int32)), 5, 2),
    (IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int64)), 20, 3),
    (IntTensorGenerator(tf.TensorSpec(shape=[5], dtype=tf.int64)), 10, 1),
    (IntTensorGenerator(tf.TensorSpec(shape=[1, 2], dtype=tf.int64)), 10, 1),
    (FloatTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.float32)), 15, 2),
    (FloatTensorGenerator(tf.TensorSpec(shape=[10], dtype=tf.float32)), 10, 1),
    (FloatTensorGenerator(tf.TensorSpec(shape=[2, 4, 6], dtype=tf.float32)), 3, 3),
    (FloatTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.float64)), 2, 1),
    (FloatTensorGenerator(tf.TensorSpec(shape=[1], dtype=tf.float64)), 10, 1),
    (FloatTensorGenerator(tf.TensorSpec(shape=[2, 4], dtype=tf.float64)), 20, 3),
    (WordTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.string)), 50, 1),
    (WordTensorGenerator(tf.TensorSpec(shape=[5], dtype=tf.string)), 40, 3),
    (WordTensorGenerator(tf.TensorSpec(shape=[2, 1], dtype=tf.string)), 20, 2),
    (BoolTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.bool)), 30, 3),
    (BoolTensorGenerator(tf.TensorSpec(shape=[5], dtype=tf.bool)), 10, 1),
    (BoolTensorGenerator(tf.TensorSpec(shape=[2, 1, 3], dtype=tf.bool)), 10, 1),
  ]
)
def test_read_from_tf_record_cache(generator, num_records, partitions):
  feature_name = "feature"
  data = [generator.generate() for _ in range(num_records)]

  mock_generator = MockGenerator(spec=generator.spec, data=data, generator_cls=type(generator))
  data_source = DataSource(
    scenario={feature_name: mock_generator},
    num_records=num_records,
    partitions=partitions
  )

  dtype = generator.spec.dtype
  data_source_cache_dir = tempfile.mkdtemp()
  count = 0
  try:
    os.environ[file_writer.TF_IO_BENCHMARK_DATA_CACHE] = data_source_cache_dir
    with ATDSWriter() as atds_writer, TFRecordWriter() as tf_writer:
      atds_path = atds_writer.write(data_source)
      tf_path = os.path.join(atds_path, os.pardir, tf_writer.hash_code())
      parser_fn = tf_writer.create_tf_example_parser_fn(data_source)
      for file_index in range(partitions):
        partition_length = len(str(partitions))
        index_name = str(file_index).zfill(partition_length)
        atds_filename = os.path.join(atds_path, f"part-{index_name}.{atds_writer.extension}")
        tf_filename = os.path.join(tf_path, f"part-{index_name}.{tf_writer.extension}")
        atds_dataset = get_dataset(atds_filename, get_features_from_data_source(atds_writer, data_source))
        atds_dataset = atds_dataset.unbatch()
        tf_dataset = tf.data.Dataset.list_files(tf_filename)
        tf_dataset = tf.data.TFRecordDataset(tf_dataset)
        tf_dataset = tf_dataset.map(parser_fn)
        for atds_record, tf_record in zip(atds_dataset, tf_dataset):
          actual = tf.cast(tf_record[feature_name], dtype)
          if dtype in [tf.float32, tf.float64]:
            tf.debugging.assert_near(actual, atds_record[feature_name], atol=1e-6)
            tf.debugging.assert_near(data[count], atds_record[feature_name], atol=1e-6)
          else:
            tf.debugging.assert_equal(actual, atds_record[feature_name])
            tf.debugging.assert_equal(data[count], atds_record[feature_name])
          count += 1
    assert count == num_records
  finally:
    del os.environ[file_writer.TF_IO_BENCHMARK_DATA_CACHE]
    shutil.rmtree(data_source_cache_dir)
