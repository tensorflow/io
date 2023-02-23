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

import pytest
import tensorflow as tf

from tensorflow_io.python.experimental.benchmark.data_source import \
  DataSource
from tensorflow_io.python.experimental.benchmark.generator.\
  tensor_generator import IntTensorGenerator, FloatTensorGenerator, \
    WordTensorGenerator
from tensorflow_io.python.experimental.benchmark.generator. \
  sparse_tensor_generator import ValueDistribution, \
    FloatSparseTensorGenerator, BoolSparseTensorGenerator, \
    WordSparseTensorGenerator, IntSparseTensorGenerator
from tensorflow_io.python.experimental.benchmark.generator. \
  varlen_tensor_generator import DimensionDistribution, \
    WordVarLenTensorGenerator, FloatVarLenTensorGenerator


@pytest.mark.parametrize(
  ["scenario", "num_records", "partitions", "expected_hash_code"], [
    ({"A": IntTensorGenerator(
       tf.TensorSpec(shape=[], dtype=tf.int32))}, 10, 1,
     "40f3d20141c3bd0766a5c067b9859a012a58ba2d2e43c2ec9a69fa477e98b8ba"),
    ({"B": FloatTensorGenerator(
       tf.TensorSpec(shape=[10], dtype=tf.float32))}, 23, 3,
     "f8d7bcb1a29f4c655a114e7346f2d3825d9817f0a308523c71c5444dc3fa27c1"),
    ({"C": WordTensorGenerator(
       tf.TensorSpec(shape=[2, 1], dtype=tf.string))}, 5, 6,
     "d769e3dc501fb73719895d5b75f7023a94e56af43ed673b5288c72c012fe71c7"),
    ({"D": FloatSparseTensorGenerator(
       tf.SparseTensorSpec(shape=[None, None], dtype=tf.float64),
       num_values=ValueDistribution.SMALL_NUM_VALUE)}, 7, 2,
     "722bf81775a65338339bbd4ed8f14470dce6f1fe5962dc621d11b82f820258f5"),
    ({"E": BoolSparseTensorGenerator(
       tf.SparseTensorSpec(shape=[2], dtype=tf.bool),
       num_values=ValueDistribution.SINGLE_VALUE)}, 99, 1,
     "f0f7defcb8bd943b909a44cd86298e5ed3d8a7aae8f146ce6ee689801eaf2397"),
    ({"F": IntSparseTensorGenerator(
       tf.SparseTensorSpec([1], tf.int32),
       num_values=ValueDistribution.LARGE_NUM_VALUE)}, 15, 2,
     "8961d67c2d41c3579eff3cbff2225a913219c5100247395700a7b1b17bae5dc5"),
    ({"G": WordSparseTensorGenerator(
       tf.SparseTensorSpec([None], tf.string),
       num_values=ValueDistribution.LARGE_NUM_VALUE)}, 15, 2,
     "a9caa46df73499280eed344f66aaabcdee304e64ce7497221629642cf589de54"),
    ({"H": WordSparseTensorGenerator(
       tf.SparseTensorSpec([None], tf.string),
       num_values=ValueDistribution.LARGE_NUM_VALUE, avg_length=7)}, 15, 2,
     "8d1924992c0604956930e36e30f46f30801029adef4d47dda45116713be6e44f"),
    ({"I": WordVarLenTensorGenerator(
       tf.SparseTensorSpec(shape=[None], dtype=tf.string),
       dim_dist=DimensionDistribution.ONE_DIM)}, 5, 6,
     "d30cdac6278b8a18072435b13b5afd556c547e9af92a717c474b5964d78ed7cc"),
    ({"J": FloatVarLenTensorGenerator(
       tf.SparseTensorSpec(shape=[None, 2], dtype=tf.float32),
       dim_dist=DimensionDistribution.TWO_DIM)}, 5, 6,
     "155d7e299e43d9c7cdcb8a04555470b4f3eb0cf75db6feaa16a487fc16b9c20a"),
    ({"K": FloatVarLenTensorGenerator(
       tf.SparseTensorSpec(shape=[101, None], dtype=tf.float32),
       dim_dist=DimensionDistribution.LARGE_DIM)}, 5, 6,
     "f00bd9c60268e136c633c05e88e62138c1df5eda6c612f16a918635321d68e6a"),
    ({
       "K": FloatVarLenTensorGenerator(
         tf.SparseTensorSpec(shape=[101, None], dtype=tf.float64),
         dim_dist=DimensionDistribution.LARGE_DIM),
       "F": IntSparseTensorGenerator(
         tf.SparseTensorSpec([1], tf.int32),
         num_values=ValueDistribution.LARGE_NUM_VALUE),
       "A": IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int32))
     }, 5, 6,
     "64b0722302638e563848baa81ed5ed53b2e780aeeb9294199dc6922776ec00fe")
  ]
)
def test_hash_equal(scenario, num_records, partitions, expected_hash_code):
  data_source = DataSource(scenario, num_records, partitions)
  assert data_source.hash_code() == expected_hash_code


@pytest.mark.parametrize(
  ["scenario", "num_records", "partitions"], [
    ({"A": IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int32))},
     100, 2),
    ({"A": IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int32))},
     10, 1),
    ({"A": IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int64))},
     100, 1),
    ({"A": IntTensorGenerator(tf.TensorSpec(shape=[1], dtype=tf.int32))},
     100, 1),
    ({"AA": IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int32))},
     100, 1),
    ({
       "A": IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int32)),
       "B": IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int32))
     }, 100, 1),
  ]
)
def test_hash_not_equal(scenario, num_records, partitions):
  data_source = DataSource(
    scenario={"A": IntTensorGenerator(tf.TensorSpec(shape=[], dtype=tf.int32))},
    num_records=100,
    partitions=1
  )

  to_compare = DataSource(scenario, num_records, partitions)
  assert data_source.hash_code() != to_compare.hash_code()


@pytest.mark.parametrize(
  ["dist1", "dist2"], [
    (ValueDistribution.SINGLE_VALUE, ValueDistribution.SMALL_NUM_VALUE),
    (ValueDistribution.SINGLE_VALUE, ValueDistribution.LARGE_NUM_VALUE),
    (ValueDistribution.SMALL_NUM_VALUE, ValueDistribution.LARGE_NUM_VALUE),
  ]
)
def test_value_dist_hash_not_equal(dist1, dist2):
  feature_name = "A"
  spec = tf.SparseTensorSpec(shape=[2, 2], dtype=tf.float64)
  num_records = 100

  data_source = DataSource(
    scenario={
      feature_name: FloatSparseTensorGenerator(spec, num_values=dist1)
    },
    num_records=num_records
  )

  to_compare = DataSource(
    scenario={
      feature_name: FloatSparseTensorGenerator(spec, num_values=dist2)
    },
    num_records=num_records
  )
  assert data_source.hash_code() != to_compare.hash_code()


@pytest.mark.parametrize(
  ["dist1", "dist2"], [
    (DimensionDistribution.ONE_DIM, DimensionDistribution.TWO_DIM),
    (DimensionDistribution.TWO_DIM, DimensionDistribution.LARGE_DIM),
    (DimensionDistribution.ONE_DIM, DimensionDistribution.LARGE_DIM),
  ]
)
def test_dim_dist_hash_not_equal(dist1, dist2):
  feature_name = "A"
  spec = tf.SparseTensorSpec(shape=[None], dtype=tf.string)
  num_records = 100

  data_source = DataSource(
    scenario={
      feature_name: WordVarLenTensorGenerator(spec, dim_dist=dist1)
    },
    num_records=num_records
  )

  to_compare = DataSource(
    scenario={
      feature_name: WordVarLenTensorGenerator(spec, dim_dist=dist2)
    },
    num_records=num_records
  )
  assert data_source.hash_code() != to_compare.hash_code()


@pytest.mark.parametrize(
  ["cls", "kwargs"], [
    (WordTensorGenerator, {"spec": tf.TensorSpec([], tf.string)}),
    (WordSparseTensorGenerator, {
       "spec": tf.SparseTensorSpec([1], tf.string),
       "num_values": ValueDistribution.SINGLE_VALUE
    }),
    (WordVarLenTensorGenerator, {
       "spec": tf.SparseTensorSpec([None], tf.string)
    }),
  ]
)
def test_avg_length_hash_not_equal(cls, kwargs):
  feature_name = "A"
  num_records = 100
  length1 = 3
  length2 = 5

  data_source = DataSource(
    scenario={
      feature_name: cls(avg_length=length1, **kwargs)
    },
    num_records=num_records
  )

  to_compare = DataSource(
    scenario={
      feature_name: cls(avg_length=length2, **kwargs)
    },
    num_records=num_records
  )
  assert data_source.hash_code() != to_compare.hash_code()
