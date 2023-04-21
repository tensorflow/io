# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
import os
import hashlib
import json
import numpy as np
import tensorflow as tf

from avro.schema import Parse as parse
from avro.datafile import DataFileWriter
from avro.io import DatumWriter

from tests.test_atds_avro.utils.file_writer import FileWriter
from tests.test_atds_avro.utils.generator.varlen_tensor_generator import (
    VarLenTensorGeneratorBase,
)
from tensorflow_io.python.experimental.atds.features import (
    DenseFeature,
    SparseFeature,
    VarlenFeature,
)


class ATDSWriter(FileWriter):
    # ATDSWriter generates Avro data from input DataSource.
    _BOOL_LIST_DTYPE = [tf.bool]
    _INT64_LIST_DTYPE = [tf.int32, tf.int64, tf.uint32, tf.uint64]
    _FLOAT_LIST_DTYPE = [tf.float32, tf.float64]
    _BYTES_LIST_DTYPE = [tf.string]
    _AVRORECORD_EXTENSION = "avro"
    _SPARSE_INDICES_KEY = "indices"
    _SPARSE_VALUES_KEY = "values"

    _DTYPE_TO_AVRO = {
        tf.float32: "float",
        tf.float64: "double",
        tf.int32: "int",
        tf.int64: "long",
        tf.bool: "boolean",
        tf.string: "bytes",
    }

    _AVRO_TO_SPARSE_TENSOR = {
        "int": "IntSparseTensor",
        "long": "LongSparseTensor",
        "double": "DoubleSparseTensor",
        "float": "FloatSparseTensor",
        "string": "StringSparseTensor",
        "bytes": "BytesSparseTensor",
        "boolean": "BoolSparseTensor",
    }

    def __init__(self, codec="null"):
        """Create a new FileWriter.

        This must be called by the constructors of subclasses.
        """
        super().__init__()
        self._codec = codec

    @property
    def extension(self):
        """Return the file extension of the written files."""
        return ATDSWriter._AVRORECORD_EXTENSION

    def hash_code(self):
        """Return the hashed code of this file writer"""
        hash_code = super().hash_code()

        m = hashlib.sha256()
        m.update(hash_code.encode())
        m.update(self._codec.encode())
        return m.hexdigest()

    def _write_to_path_from_cached_data(self, dir_path, data_source, dataset):
        if os.path.exists(dir_path):
            return dir_path
        os.makedirs(dir_path)

        scenario = data_source.scenario
        schema = parse(self.scenario_to_avro_schema(scenario))

        filenames_to_num_records = self._get_filenames_to_num_records(data_source)
        iterator = iter(dataset)
        for filename in sorted(filenames_to_num_records):
            file_path = os.path.join(dir_path, filename)
            num_records = filenames_to_num_records[filename]

            with open(file_path, "wb") as out:
                file_writer = DataFileWriter(
                    out, DatumWriter(), schema, codec=self._codec
                )
                for _ in range(num_records):
                    features = {}
                    record = iterator.get_next()
                    for feature_name in record:
                        generator = data_source.scenario[feature_name]
                        # tf.Example only supports tf.float32, tf.int64, and tf.string
                        # For other dtypes, cast feature into its original dtype.
                        actual = tf.cast(record[feature_name], generator.spec.dtype)
                        self._add_tensor_to_features(
                            generator, feature_name, actual, features
                        )
                    file_writer.append(features)
                file_writer.close()

    def _write_to_path(self, dir_path, data_source):
        """Generate benchmark data and write the data under the given path.

        Args:
          dir_path: A str path to write files to.
          data_source: A DataSource objects.

        Raises:
          NotImplementedError: If subclass does not overload the function.
        """
        scenario = data_source.scenario
        schema = parse(self.scenario_to_avro_schema(scenario))

        filenames_to_num_records = self._get_filenames_to_num_records(data_source)
        for filename in sorted(filenames_to_num_records):
            file_path = os.path.join(dir_path, filename)
            num_records = filenames_to_num_records[filename]

            with open(file_path, "wb") as out:
                file_writer = DataFileWriter(
                    out, DatumWriter(), schema, codec=self._codec
                )
                for _ in range(num_records):
                    features = {}
                    for feature_name in scenario:
                        generator = scenario[feature_name]
                        tensor = generator.generate()
                        self._add_tensor_to_features(
                            generator, feature_name, tensor, features
                        )
                    file_writer.append(features)
                file_writer.close()

    def _add_tensor_to_features(self, generator, feature_name, tensor, features):
        spec = generator.spec
        if isinstance(spec, tf.TensorSpec):
            self._add_dense_tensor_to_features(feature_name, tensor, features)
        elif isinstance(spec, tf.SparseTensorSpec):
            if issubclass(generator.get_generator_cls(), VarLenTensorGeneratorBase):
                self._add_dense_tensor_to_features(
                    feature_name, tf.sparse.to_dense(tensor), features
                )
            else:
                self._add_sparse_tensor_to_features(feature_name, tensor, features)
        else:
            raise TypeError(f"Spec {spec} is not supported in ATDSWriter")

    def scenario_to_avro_schema(self, scenario):
        """
        Goes through a scenario to convert it to an avro schema

        """
        schema = {"type": "record", "name": "row", "fields": []}
        for feature_name in scenario:
            generator = scenario[feature_name]
            spec = generator.spec
            if isinstance(spec, tf.TensorSpec):
                self._add_dense_feature_schema(feature_name, spec, schema)
            elif isinstance(spec, tf.SparseTensorSpec):
                if issubclass(generator.get_generator_cls(), VarLenTensorGeneratorBase):
                    self._add_dense_feature_schema(feature_name, spec, schema)
                else:
                    self._add_sparse_feature_schema(feature_name, spec, "long", schema)
        schema_str = json.dumps(schema)
        return schema_str

    def _map_tensor_dtype_to_avro_dtype(self, dtype):
        """This function maps tensor dtype into the python type suppored by avro."""
        if dtype in ATDSWriter._DTYPE_TO_AVRO:
            return ATDSWriter._DTYPE_TO_AVRO[dtype]
        else:
            raise TypeError(f"Type {dtype} is not supported in Avro.")

    def _add_dense_feature_schema(self, name, spec, schema):
        schema["fields"].append(
            {
                "name": name,
                "type": self._add_dense_feature_type(
                    spec.dtype, len(spec.shape.as_list())
                ),
            }
        )

    def _add_dense_feature_type(self, dtype, rank):
        # if scalar then convert tf type to python type name
        avro_type = self._map_tensor_dtype_to_avro_dtype(dtype)
        if rank == 0:
            return avro_type
        else:
            return {
                "type": "array",
                "items": self._add_dense_feature_type(dtype, rank - 1),
            }

    def _infer_sparse_tensor_type(self, dtype):
        value_avro_type = self._map_tensor_dtype_to_avro_dtype(dtype)
        return ATDSWriter._AVRO_TO_SPARSE_TENSOR[value_avro_type]

    def _add_sparse_feature_schema(self, name, spec, indices_avro_type, schema):
        schema["fields"].append(
            {
                "name": name,
                "type": {
                    "type": "record",
                    "name": name + "_" + self._infer_sparse_tensor_type(spec.dtype),
                    "fields": self._add_sparse_feature_fields(
                        spec.dtype, indices_avro_type, len(spec.shape.as_list())
                    ),
                },
            }
        )

    def _add_sparse_feature_fields(self, dtype, indices_avro_type, rank):
        value_avro_type = self._map_tensor_dtype_to_avro_dtype(dtype)
        fields = []
        for dim in range(rank):
            indices_name_at_dim = ATDSWriter._SPARSE_INDICES_KEY + str(dim)
            fields.append(
                {
                    "name": indices_name_at_dim,
                    "type": {"type": "array", "items": indices_avro_type},
                }
            )
        value_field = {
            "name": ATDSWriter._SPARSE_VALUES_KEY,
            "type": {"type": "array", "items": value_avro_type},
        }
        fields.append(value_field)
        return fields

    def _get_flat_value(self, tensor):
        rank = len(tensor.shape.as_list())
        value = tensor.numpy()
        if (
            tensor.dtype == tf.string and rank == 0
        ):  # There is no tolist() method for bytes or string
            return value
        return value.tolist()

    def _add_dense_tensor_to_features(self, name, tensor, features):
        features[name] = self._get_flat_value(tensor)

    def _add_sparse_tensor_to_features(self, name, tensor, features):
        rank = len(tensor.shape.as_list())
        indices = tensor.indices.numpy()  # indices tensor must be a 2D array
        # Split indices array along the second dimension so that the split arrays
        # contain the indices for separate dimension. For example,
        # indices = [[0, 1], [2, 3], [4, 5]] =>
        #   indices_at_dim0 = [[0], [2], [4]] and
        #   indices_at_dim1 = [[1], [3], [5]].
        split_indices = np.split(indices, rank, axis=1)
        features[name] = {}
        for dim in range(rank):
            indices_name_at_dim = ATDSWriter._SPARSE_INDICES_KEY + str(dim)
            # convert indices to 1d array
            features[name][indices_name_at_dim] = split_indices[dim].flatten().tolist()

        features[name][ATDSWriter._SPARSE_VALUES_KEY] = self._get_flat_value(
            tensor.values
        )

    def _get_atds_feature(self, generator):
        """Build tf.Example parsing config

        Args:
          generator: A TensorGenerator, SparseTensorGenerator, or VarLenTensorGenerator for generating data.

        Returns:
          DenseFeature, SparseFeature, or VarlenFeature

        Raises:
          TypeError: if spec is not tf.TensorSpec or tf.SparseTensorSpec.
        """
        spec = generator.spec
        if isinstance(spec, tf.TensorSpec):
            return DenseFeature(shape=spec.shape.as_list(), dtype=spec.dtype)
        elif isinstance(spec, tf.SparseTensorSpec):
            if issubclass(generator.get_generator_cls(), VarLenTensorGeneratorBase):
                atds_shape = [
                    -1 if dim == None else dim for dim in spec.shape.as_list()
                ]
                return VarlenFeature(shape=atds_shape, dtype=spec.dtype)
            else:
                return SparseFeature(shape=spec.shape.as_list(), dtype=spec.dtype)
        else:
            raise TypeError(f"Spec {spec} is not supported in ATDSWriter.")
