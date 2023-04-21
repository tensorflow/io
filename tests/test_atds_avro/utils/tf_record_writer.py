# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""TFRecordWriter"""

import os
from tests.test_atds_avro.utils.generator.tensor_generator import (
    TensorGeneratorBase,
)
from tests.test_atds_avro.utils.generator.sparse_tensor_generator import (
    SparseTensorGeneratorBase,
)
from tests.test_atds_avro.utils.generator.varlen_tensor_generator import (
    VarLenTensorGeneratorBase,
)
import numpy as np
import tensorflow as tf

from tests.test_atds_avro.utils.file_writer import FileWriter


class TFRecordWriter(FileWriter):
    """File writer for TFRecord dataset.

    TFRecordWriter serializes tensors in tf.Example schema and write them
    into files in TFRecord format. The written file can be loaded with
    tf.data.TFRecordDataset.
    """

    # The TFRecord file extension.
    _TFRECORD_EXTENSION = "tfrecords"

    # TFRecord Dataset reads TFRecord data serialized in tf.Example schema.
    # tf.Example only supported three dtypes i.e. int64, float32, and bytes.
    # See https://www.tensorflow.org/tutorials/load_data/tfrecord#data_types_for_tftrainexample
    # The lists below are used to map tensor dtype into supported dtype in
    # tf.Example. For example, tf.bool will be mapped to int64.
    _INT64_LIST_DTYPE = [tf.bool, tf.int32, tf.int64, tf.uint32, tf.uint64]
    _FLOAT_LIST_DTYPE = [tf.float32, tf.float64]
    _BYTES_LIST_DTYPE = [tf.string]

    # Sparse tensor is composed of many 1D dense tensors in tf.Example.
    # The suffix is used to name these dense tensors given the sparse
    # tensor name. For example, a 2D sparse tensor 'feature' was represented
    # with three dense tensors with name 'feature/indices0', 'feature/indices1',
    # and 'feature/values'.
    _SPARSE_INDICES_SUFFIX = "/indices"
    _SPARSE_VALUES_SUFFIX = "/values"

    def __init__(self):
        """Create a new TFRecordWriter"""
        super().__init__()

    @property
    def extension(self):
        """Return the file extension of the written files."""
        return TFRecordWriter._TFRECORD_EXTENSION

    def _write_to_path(self, dir_path, data_source):
        """Generate data based on the data_source and write
        files under the given path."""
        scenario = data_source.scenario

        filenames_to_num_records = self._get_filenames_to_num_records(data_source)
        for filename in sorted(filenames_to_num_records):
            file_path = os.path.join(dir_path, filename)
            num_records = filenames_to_num_records[filename]
            with tf.io.TFRecordWriter(file_path) as file_writer:
                for _ in range(num_records):
                    features = {}

                    for feature_name in scenario:
                        generator = scenario[feature_name]
                        tensor = generator.generate()
                        self._add_tensor_to_features(
                            generator, feature_name, tensor, features
                        )
                    record_bytes = tf.train.Example(
                        features=tf.train.Features(feature=features)
                    ).SerializeToString()
                    file_writer.write(record_bytes)

    def _write_to_path_from_cached_data(self, dir_path, data_source, dataset):
        if os.path.exists(dir_path):
            return dir_path
        os.makedirs(dir_path)

        scenario = data_source.scenario
        filenames_to_num_records = self._get_filenames_to_num_records(data_source)
        iterator = iter(dataset)
        for filename in sorted(filenames_to_num_records):
            file_path = os.path.join(dir_path, filename)
            num_records = filenames_to_num_records[filename]

            with tf.io.TFRecordWriter(file_path) as file_writer:
                for _ in range(num_records):
                    features = {}
                    record = iterator.get_next()

                    for feature_name in scenario:
                        generator = scenario[feature_name]
                        feature = record[feature_name]
                        self._add_tensor_to_features(
                            generator, feature_name, feature, features
                        )
                    record_bytes = tf.train.Example(
                        features=tf.train.Features(feature=features)
                    ).SerializeToString()
                    file_writer.write(record_bytes)

    def _add_dense_tensor_to_features(self, name, value, dtype, features):
        """Wrap value np.array into tf.train.Feature and add it into features."""
        if np.isscalar(value):
            value = [value]  # Convert scalar into a list.
        else:
            value = value.flatten()

        example_dtype = self._map_tensor_dtype_to_example_dtype(dtype)
        if example_dtype is tf.int64:
            features[name] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=value)
            )
        elif example_dtype is tf.float32:
            features[name] = tf.train.Feature(
                float_list=tf.train.FloatList(value=value)
            )
        elif example_dtype is tf.string:
            features[name] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=value)
            )
        else:
            raise TypeError(f"Dtype {dtype} is not supported in tf.Example.")

    def _map_tensor_dtype_to_example_dtype(self, dtype):
        """As tf.Example only supports tf.float32, tf.int64, and tf.string dtype.
        This function maps tensor dtype into the dtype supported by tf.Example."""
        if dtype in TFRecordWriter._INT64_LIST_DTYPE:
            return tf.int64
        elif dtype in TFRecordWriter._FLOAT_LIST_DTYPE:
            return tf.float32
        elif dtype in TFRecordWriter._BYTES_LIST_DTYPE:
            return tf.string
        else:
            raise TypeError(f"Dtype {dtype} is not supported in tf.Example.")

    def _add_sparse_tensor_to_features(self, name, tensor, features):
        indices_name = name + TFRecordWriter._SPARSE_INDICES_SUFFIX
        values_name = name + TFRecordWriter._SPARSE_VALUES_SUFFIX

        rank = len(tensor.shape.as_list())
        indices = tensor.indices.numpy()  # indices tensor must be a 2D array
        # Split indices array along the second dimension so that the split arrays
        # contain the indices for separate dimension. For example,
        # indices = [[0, 1], [2, 3], [4, 5]] =>
        #   indices_at_dim0 = [[0], [2], [4]] and
        #   indices_at_dim1 = [[1], [3], [5]].
        split_indices = np.split(indices, rank, axis=1)
        for dim in range(rank):
            indices_name_at_dim = indices_name + str(dim)
            self._add_dense_tensor_to_features(
                name=indices_name_at_dim,
                value=split_indices[dim],
                dtype=tensor.indices.dtype,
                features=features,
            )

        self._add_dense_tensor_to_features(
            name=values_name,
            value=tensor.values.numpy(),
            dtype=tensor.values.dtype,
            features=features,
        )

    def _add_tensor_to_features(self, generator, feature_name, tensor, features):
        spec = generator.spec
        if isinstance(spec, tf.TensorSpec):
            self._add_dense_tensor_to_features(
                feature_name, tensor.numpy(), tensor.dtype, features
            )
        elif isinstance(spec, tf.SparseTensorSpec):
            if (
                issubclass(generator.get_generator_cls(), VarLenTensorGeneratorBase)
                and spec.shape.rank == 1
            ):
                self._add_dense_tensor_to_features(
                    feature_name,
                    tf.sparse.to_dense(tensor).numpy(),
                    tensor.dtype,
                    features,
                )
            elif (
                issubclass(generator.get_generator_cls(), SparseTensorGeneratorBase)
                and spec.shape.is_fully_defined()
            ):
                self._add_sparse_tensor_to_features(feature_name, tensor, features)
            else:
                raise ValueError(
                    "SparseTensorSpec shape must be either a 1D varlen tensor from VarLenTensorGenerator "
                    f"or fully defined sparse tensor from SparseTensorGenerator. Found {spec}"
                )
        else:
            raise TypeError(f"Spec {spec} is not supported in TFRecordWriter")

    def create_tf_example_parser_fn(self, data_source, with_batch=False):
        """Create tf.Example parser function based on the data_source.

        The parser function can be used for parsing tf.Example.
        Example usage:

        ``` python
        data_source = DataSource(...)
        with TFRecordWriter() as writer:
            dir_path = writer.write(data_source)
            parser_fn = writer.create_tf_example_parser_fn(data_source)

            pattern = os.path.join(dir_path, f"*.{writer.extension}")
            dataset = tf.data.Dataset.list_files(pattern)
            dataset = tf.data.TFRecordDataset(dataset)
            dataset = dataset.map(parser_fn)
        ```

        Args:
          data_source: A DataSource object describe the format of the data.
          with_batch: True if the parser function should take a number of
                      serialized tf.Example proto. Default is false.

        Returns:
          A callable function that takes serialized tf.Example proto as input,
          and returns the parsed tensor dict.
        """
        scenario = data_source.scenario
        feature_description = {
            name: self._build_tf_example_parsing_config(name, scenario[name])
            for name in scenario
        }

        if with_batch:

            def _batch_examples_parser_fn(example_proto):
                return tf.io.parse_example(example_proto, feature_description)

            return _batch_examples_parser_fn

        def _single_example_parser_fn(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)

        return _single_example_parser_fn

    def _build_tf_example_parsing_config(self, name, generator):
        """Build tf.Example parsing config

        Args:
          name: A str feature name.
          generator: Generator for this tensor.

        Returns:
          tf.io.FixedLenFeature if generator is TensorGenerator.
          tf.io.SparseFeature if generator is SparseTensorGenerator.
          tf.io.VarLenFeature if generator is VarlenTensorGenerator.

        Raises:
          TypeError: if generator is not TensorGenerator, SparseTensorGenerator, or VarlenTensorGenerator.
        """
        spec = generator.spec
        example_dtype = self._map_tensor_dtype_to_example_dtype(spec.dtype)
        if isinstance(spec, tf.TensorSpec):
            return tf.io.FixedLenFeature(shape=spec.shape, dtype=example_dtype)
        elif isinstance(spec, tf.SparseTensorSpec):
            if (
                issubclass(generator.get_generator_cls(), VarLenTensorGeneratorBase)
                and spec.shape.rank == 1
            ):
                return tf.io.VarLenFeature(dtype=example_dtype)
            elif (
                issubclass(generator.get_generator_cls(), SparseTensorGeneratorBase)
                and spec.shape.is_fully_defined()
            ):
                index_name = name + TFRecordWriter._SPARSE_INDICES_SUFFIX
                rank = len(spec.shape)
                index_key = [f"{index_name}{dim}" for dim in range(rank)]
                value_key = name + TFRecordWriter._SPARSE_VALUES_SUFFIX
                return tf.io.SparseFeature(
                    index_key=index_key,
                    value_key=value_key,
                    dtype=example_dtype,
                    size=spec.shape,
                )
            else:
                raise ValueError(
                    "SparseTensorSpec shape must be either a 1D varlen tensor from VarLenTensorGenerator "
                    f"or fully defined sparse tensor from SparseTensorGenerator. Found {spec}"
                )
        else:
            raise TypeError(f"Spec {spec} is not supported in TFRecordWriter.")
