# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

# From CSV dataset
# https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/data/experimental/ops/readers.py

# Parse example dataset
# https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/data/experimental/ops/parsing_ops.py

# For tf export use examples see
# sql_dataset_test_base.py for use and readers.py for definition
"""AVRODataset"""

import collections
import re

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.experimental.parse_avro_ops import (
    construct_tensors_for_composite_features,
)

# Note: I've hidden the dataset because it does not apply the mapping for
# sparse tensors
# Such mapping is not possible inside the dataset, rather it needs to happen
# through a map on the output of the dataset which is a map of keys to tensors
# This can be changed when eager mode is the default and the only mode supported
class _AvroDataset(tf.data.Dataset):
    """A `DatasetSource` that reads and parses Avro records from files."""

    def __init__(
        self,
        filenames,
        features,
        reader_schema,
        batch_size,
        drop_remainder,
        num_parallel_calls,
        input_stream_buffer_size,
        avro_data_buffer_size,
    ):

        self._filenames = tf.convert_to_tensor(filenames, tf.string, name="filenames")
        self._features = _AvroDataset._build_keys_for_sparse_features(features)
        self._reader_schema = reader_schema
        self._batch_size = tf.convert_to_tensor(
            batch_size, dtype=tf.int64, name="batch_size"
        )
        self._drop_remainder = tf.convert_to_tensor(
            drop_remainder, dtype=tf.bool, name="drop_remainder"
        )
        self._num_parallel_calls = num_parallel_calls
        self._input_stream_buffer_size = input_stream_buffer_size
        self._avro_data_buffer_size = avro_data_buffer_size

        # Copied from _ParseExampleDataset from data/experimental/ops/parsing_ops.py
        (
            sparse_keys,
            sparse_types,
            sparse_dense_shapes,
            dense_keys,
            dense_types,
            dense_defaults,
            dense_shapes,
        ) = _AvroDataset._features_to_raw_params(
            self._features,
            [tf.io.VarLenFeature, tf.io.SparseFeature, tf.io.FixedLenFeature],
        )

        (
            _,
            dense_defaults_vec,
            sparse_keys,
            sparse_types,
            dense_keys,
            dense_shapes,
            _,
        ) = _AvroDataset._process_raw_parameters(
            None,
            dense_defaults,
            sparse_keys,
            sparse_types,
            dense_keys,
            dense_types,
            dense_shapes,
        )

        self._sparse_keys = sparse_keys
        self._sparse_types = sparse_types
        self._dense_keys = dense_keys
        self._dense_defaults = dense_defaults_vec
        self._dense_types = dense_types

        output_shapes = dict(
            zip(
                self._dense_keys + self._sparse_keys, dense_shapes + sparse_dense_shapes
            )
        )
        output_types = dict(
            zip(
                self._dense_keys + self._sparse_keys,
                self._dense_types + self._sparse_types,
            )
        )
        output_classes = dict(
            zip(
                self._dense_keys + self._sparse_keys,
                [tf.Tensor for _ in range(len(self._dense_defaults))]
                + [tf.sparse.SparseTensor for _ in range(len(self._sparse_keys))],
            )
        )

        self._element_spec = _AvroDataset._convert_legacy_structure(
            output_types, output_shapes, output_classes
        )

        constant_drop_remainder = tf.get_static_value(self._drop_remainder)
        # pylint: disable=protected-access
        if constant_drop_remainder:
            # NOTE(mrry): `constant_drop_remainder` may be `None` (unknown statically)
            # or `False` (explicitly retaining the remainder).
            # pylint: disable=g-long-lambda
            self._element_spec = tf.nest.map_structure(
                lambda component_spec: component_spec._batch(
                    tf.get_static_value(self._batch_size)
                ),
                self._element_spec,
            )
        else:
            self._element_spec = tf.nest.map_structure(
                lambda component_spec: component_spec._batch(None), self._element_spec
            )

        # With batch dimension
        self._dense_shapes = [
            spec.shape
            for spec in tf.nest.flatten(self._element_spec)
            if isinstance(spec, tf.TensorSpec)
        ]

        variant_tensor = core_ops.io_avro_dataset(
            filenames=self._filenames,  # pylint: disable=protected-access
            batch_size=self._batch_size,
            drop_remainder=self._drop_remainder,
            dense_defaults=self._dense_defaults,
            input_stream_buffer_size=self._input_stream_buffer_size,
            avro_data_buffer_size=self._avro_data_buffer_size,
            reader_schema=self._reader_schema,
            sparse_keys=self._sparse_keys,
            dense_keys=self._dense_keys,
            sparse_types=self._sparse_types,
            dense_shapes=self._dense_shapes,
            **self._flat_structure
        )

        super().__init__(variant_tensor)

    def _inputs(self):
        return []

    @staticmethod
    # copied from https://github.com/tensorflow/tensorflow/blob/
    # 858bd4506a8b390fc5b2dcbeb3057f4a92e8a1e2/tensorflow/python/data/util/structure.py#L119
    def _convert_legacy_structure(output_types, output_shapes, output_classes):
        """convert_legacy_structure"""
        flat_types = tf.nest.flatten(output_types)
        flat_shapes = tf.nest.flatten(output_shapes)
        flat_classes = tf.nest.flatten(output_classes)
        flat_ret = []
        for flat_type, flat_shape, flat_class in zip(
            flat_types, flat_shapes, flat_classes
        ):
            if isinstance(flat_class, tf.TypeSpec):
                flat_ret.append(flat_class)
            elif issubclass(flat_class, tf.sparse.SparseTensor):
                flat_ret.append(tf.SparseTensorSpec(flat_shape, flat_type))
            elif issubclass(flat_class, tf.Tensor):
                flat_ret.append(tf.TensorSpec(flat_shape, flat_type))
            elif issubclass(flat_class, tf.TensorArray):
                # We sneaked the dynamic_size and infer_shape into the legacy shape.
                flat_ret.append(
                    tf.TensorArraySpec(
                        flat_shape[2:],
                        flat_type,
                        dynamic_size=tf.compat.dimension_value(flat_shape[0]),
                        infer_shape=tf.compat.dimension_value(flat_shape[1]),
                    )
                )
            else:
                # NOTE(mrry): Since legacy structures produced by iterators only
                # comprise Tensors, SparseTensors, and nests, we do not need to
                # support all structure types here.
                raise TypeError(
                    "Could not build a structure for output class {!r}".format(
                        flat_class
                    )
                )

        return tf.nest.pack_sequence_as(output_classes, flat_ret)

    @property
    def element_spec(self):
        return self._element_spec

    @staticmethod
    def _build_keys_for_sparse_features(features):
        """
        Builds the fully qualified names for keys of sparse features.

        :param features:  A map of features with keys to TensorFlow features.

        :return: A map of features where for the sparse feature the 'index_key'
                 and the 'value_key' have been expanded properly for the parser
                 in the native code.
        """

        def resolve_key(parser_key, index_or_value_key):
            if not index_or_value_key.startswith("@"):
                return parser_key + "[*]." + index_or_value_key
            return index_or_value_key[1:]

        def resolve_index_key(key_, index_key):
            if isinstance(index_key, list):
                return [resolve_key(key_, index_key_) for index_key_ in index_key]
            return resolve_key(key_, index_key)

        if features:
            # NOTE: We iterate over sorted keys to keep things deterministic.
            for key in sorted(features.keys()):
                feature = features[key]
                if isinstance(feature, tf.io.SparseFeature):
                    features[key] = tf.io.SparseFeature(
                        index_key=resolve_index_key(key, feature.index_key),
                        value_key=resolve_key(key, feature.value_key),
                        dtype=feature.dtype,
                        size=feature.size,
                        already_sorted=feature.already_sorted,
                    )
        return features

    # Pulled this method from tensorflow/python/ops/parsing_ops.py
    # here to customize the handling of default values because
    # we have
    # - more types
    # - handling had to change because we don't have a batch dimension when
    #   calling this method
    @staticmethod
    def _process_raw_parameters(
        names,
        dense_defaults,
        sparse_keys,
        sparse_types,
        dense_keys,
        dense_types,
        dense_shapes,
    ):
        """Process raw parameters to params used by `gen_parsing_ops`.
        Args:
          names: A vector (1-D Tensor) of strings (optional), the names of
            the serialized protos.
          dense_defaults: A dict mapping string keys to `Tensor`s.
            The keys of the dict must match the dense_keys of the feature.
          sparse_keys: A list of string keys in the examples' features.
            The results for these keys will be returned as `SparseTensor` objects.
          sparse_types: A list of `DTypes` of the same length as `sparse_keys`.
            Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
            and `tf.string` (`BytesList`) are supported.
          dense_keys: A list of string keys in the examples' features.
            The results for these keys will be returned as `Tensor`s
          dense_types: A list of DTypes of the same length as `dense_keys`.
            Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
            and `tf.string` (`BytesList`) are supported.
          dense_shapes: A list of tuples with the same length as `dense_keys`.
            The shape of the data for each dense feature referenced by `dense_keys`.
            Required for any input tensors identified by `dense_keys`.  Must be
            either fully defined, or may contain an unknown first dimension.
            An unknown first dimension means the feature is treated as having
            a variable number of blocks, and the output shape along this dimension
            is considered unknown at graph build time.  Padding is applied for
            minibatch elements smaller than the maximum number of blocks for the
            given feature along this dimension.
        Returns:
          Tuple of `names`, `dense_defaults_vec`, `sparse_keys`, `sparse_types`,
          `dense_keys`, `dense_shapes`.
        Raises:
          ValueError: If sparse and dense key sets intersect, or input lengths do not
            match up.
        """
        names = [] if names is None else names
        dense_defaults = (
            collections.OrderedDict() if dense_defaults is None else dense_defaults
        )
        sparse_keys = [] if sparse_keys is None else sparse_keys
        sparse_types = [] if sparse_types is None else sparse_types
        dense_keys = [] if dense_keys is None else dense_keys
        dense_types = [] if dense_types is None else dense_types
        dense_shapes = [[]] * len(dense_keys) if dense_shapes is None else dense_shapes

        num_dense = len(dense_keys)
        num_sparse = len(sparse_keys)

        if len(dense_shapes) != num_dense:
            raise ValueError(
                "len(dense_shapes) != len(dense_keys): %d vs. %d"
                % (len(dense_shapes), num_dense)
            )
        if len(dense_types) != num_dense:
            raise ValueError(
                "len(dense_types) != len(num_dense): %d vs. %d"
                % (len(dense_types), num_dense)
            )
        if len(sparse_types) != num_sparse:
            raise ValueError(
                "len(sparse_types) != len(sparse_keys): %d vs. %d"
                % (len(sparse_types), num_sparse)
            )
        if num_dense + num_sparse == 0:
            raise ValueError("Must provide at least one sparse key or dense key")
        if not set(dense_keys).isdisjoint(set(sparse_keys)):
            raise ValueError(
                "Dense and sparse keys must not intersect; intersection: %s"
                % set(dense_keys).intersection(set(sparse_keys))
            )

        # Convert dense_shapes to TensorShape object.
        dense_shapes = [tf.TensorShape(shape) for shape in dense_shapes]

        dense_defaults_vec = []
        for i, key in enumerate(dense_keys):
            default_value = dense_defaults.get(key)
            dense_shape = dense_shapes[i]

            # Whenever the user did not provide a default, set it

            # ************* START difference: This part is different from the originally copied code ***************
            if default_value is None:
                if dense_types[i] == tf.string:
                    default_value = ""
                elif dense_types[i] == tf.bool:
                    default_value = False
                else:  # Should be numeric type
                    default_value = 0
                default_value = tf.convert_to_tensor(
                    default_value, dtype=dense_types[i]
                )
            elif not isinstance(default_value, tf.Tensor):
                key_name = "key_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
                default_value = tf.convert_to_tensor(
                    default_value, dtype=dense_types[i], name=key_name
                )
                # If we have a shape and the first dimension is not None
                if dense_shape.rank and dense_shape.dims[0].value:
                    default_value = tf.reshape(default_value, dense_shape)
            # ************* END difference: This part is different from the originally copied code *****************
            dense_defaults_vec.append(default_value)

        # Finally, convert dense_shapes to TensorShapeProto
        dense_shapes_as_proto = [shape.as_proto() for shape in dense_shapes]

        return (
            names,
            dense_defaults_vec,
            sparse_keys,
            sparse_types,
            dense_keys,
            dense_shapes_as_proto,
            dense_shapes,
        )

    @staticmethod
    def __check_not_empty(instance, missing_str, for_str):
        """if instance is empty, raise error """
        if not instance:
            raise ValueError("Missing {} for {}.".format(missing_str, for_str))

    # Pulled this method from tensorflow/python/ops/parsing_ops.py
    # here to customize dense shape handling of sparse tensors, here we
    # - assume the shape parameter is set by the user
    # - the shape is used for the index
    # - removed support for FixedLenSequenceFeature
    @staticmethod
    def _features_to_raw_params(features, types):
        """Split feature tuples into raw params used by `gen_parsing_ops`.

        Args:
          features: A `dict` mapping feature keys to objects of a type in `types`.
          types: Type of features to allow, among `FixedLenFeature`, `VarLenFeature`,
            `SparseFeature`, and `FixedLenSequenceFeature`.

        Returns:
          Tuple of `sparse_keys`, `sparse_types`, `dense_keys`, `dense_types`,
            `dense_defaults`, `dense_shapes`.

        Raises:
          ValueError: if `features` contains an item not in `types`, or an invalid
              feature.
        """
        sparse_keys = []
        sparse_types = []
        sparse_dense_shapes = []
        dense_keys = []
        dense_types = []
        # When the graph is built twice, multiple dense_defaults in a normal dict
        # could come out in different orders. This will fail the _e2e_test which
        # expects exactly the same graph.
        # OrderedDict which preserves the order can solve the problem.
        dense_defaults = collections.OrderedDict()
        dense_shapes = []
        if features:
            # NOTE: We iterate over sorted keys to keep things deterministic.
            for key in sorted(features.keys()):
                feature = features[key]
                if isinstance(feature, tf.io.VarLenFeature):
                    _AvroDataset.__handle_varlen_feature(
                        feature,
                        key,
                        sparse_dense_shapes,
                        sparse_keys,
                        sparse_types,
                        types,
                    )
                elif isinstance(feature, tf.io.SparseFeature):
                    _AvroDataset.__handle_sparse_feature(
                        feature,
                        key,
                        sparse_dense_shapes,
                        sparse_keys,
                        sparse_types,
                        types,
                    )
                elif isinstance(feature, tf.io.FixedLenFeature):
                    _AvroDataset.__handle_fixedlen_feature(
                        dense_defaults,
                        dense_keys,
                        dense_shapes,
                        dense_types,
                        feature,
                        key,
                        types,
                    )
                else:
                    raise ValueError("Invalid feature {}:{}.".format(key, feature))
        return (
            sparse_keys,
            sparse_types,
            sparse_dense_shapes,
            dense_keys,
            dense_types,
            dense_defaults,
            dense_shapes,
        )

    @staticmethod
    def __handle_fixedlen_feature(
        dense_defaults, dense_keys, dense_shapes, dense_types, feature, key, types
    ):
        """handle_fixedlen_feature"""
        if tf.io.FixedLenFeature not in types:
            raise ValueError("Unsupported FixedLenFeature {}.".format(feature))
        _AvroDataset.__check_not_empty(feature.dtype, "type", "feature " + key)
        if feature.shape is None:
            raise ValueError("Missing shape for feature %s." % key)
        dense_keys.append(key)
        dense_shapes.append(feature.shape)
        dense_types.append(feature.dtype)
        if feature.default_value is not None:
            dense_defaults[key] = feature.default_value

    @staticmethod
    def __handle_sparse_feature(
        feature, key, sparse_dense_shapes, sparse_keys, sparse_types, types
    ):
        """handle_sparse_feature"""
        if tf.io.SparseFeature not in types:
            raise ValueError("Unsupported SparseFeature {}.".format(feature))
        _AvroDataset.__check_not_empty(
            feature.index_key, "index_key", "SparseFeature " + key
        )
        _AvroDataset.__check_not_empty(
            feature.value_key, "value_key", "SparseFeature " + key
        )
        _AvroDataset.__check_not_empty(feature.dtype, "type", "feature " + key)
        _AvroDataset.__check_not_empty(feature.size, "size", "feature " + key)
        index_keys = feature.index_key
        if isinstance(index_keys, str):
            index_keys = [index_keys]
        elif len(index_keys) > 1:
            tf.compat.v1.logging.warning(
                "SparseFeature is a complicated feature config "
                "and should only be used after careful "
                "consideration of VarLenFeature."
            )
        for index_key in sorted(index_keys):
            if index_key in sparse_keys:
                dtype = sparse_types[sparse_keys.index(index_key)]
                if dtype != tf.int64:
                    raise ValueError(
                        "Conflicting type %s vs int64 for feature %s."
                        % (dtype, index_key)
                    )
            else:
                sparse_keys.append(index_key)
                sparse_types.append(tf.int64)
                sparse_dense_shapes.append(feature.size)
        if feature.value_key in sparse_keys:
            dtype = sparse_types[sparse_keys.index(feature.value_key)]
            if dtype != feature.dtype:
                raise ValueError(
                    "Conflicting type %s vs %s for feature %s."
                    % (dtype, feature.dtype, feature.value_key)
                )
        else:
            sparse_keys.append(feature.value_key)
            sparse_types.append(feature.dtype)
            sparse_dense_shapes.append(None)  # Unknown and variable length

    @staticmethod
    def __handle_varlen_feature(
        feature, key, sparse_dense_shapes, sparse_keys, sparse_types, types
    ):
        """handle_varlen_feature"""
        if tf.io.VarLenFeature not in types:
            raise ValueError("Unsupported VarLenFeature {}.".format(feature))
        _AvroDataset.__check_not_empty(feature.dtype, "type", "feature " + key)
        sparse_keys.append(key)
        sparse_types.append(feature.dtype)
        sparse_dense_shapes.append(None)


def make_avro_dataset(
    filenames,
    reader_schema,
    features,
    batch_size,
    num_epochs,
    num_parallel_calls=2,
    label_keys=None,
    input_stream_buffer_size=16 * 1024,
    avro_data_buffer_size=256,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
    num_parallel_reads=1,
):
    """Makes an avro dataset.

    Reads from avro files and parses the contents into tensors.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      reader_schema: A `tf.string` scalar for schema resolution.

      features: Is a map of keys that describe a single entry or sparse vector
                in the avro record and map that entry to a tensor. The syntax
                is as follows:

                features = {'my_meta_data.size':
                            tf.FixedLenFeature([], tf.int64)}

                Select the 'size' field from a record metadata that is in the
                field 'my_meta_data'. In this example we assume that the size is
                encoded as a long in the Avro record for the metadata.


                features = {'my_map_data['source'].ip_addresses':
                            tf.VarLenFeature([], tf.string)}

                Select the 'ip_addresses' for the 'source' key in the map
                'my_map_data'. Notice we assume that IP addresses are encoded as
                strings in this example.


                features = {'my_friends[1].first_name':
                            tf.FixedLenFeature([], tf.string)}

                Select the 'first_name' for the second friend with index '1'.
                This assumes that all of your data has a second friend. In
                addition, we assume that all friends have only one first name.
                For this reason we chose a 'FixedLenFeature'.


                features = {'my_friends[*].first_name':
                            tf.VarLenFeature([], tf.string)}

                Selects all first_names in each row. For this example we use the
                wildcard '*' to indicate that we want to select all 'first_name'
                entries from the array.

                features = {'sparse_features':
                            tf.SparseFeature(index_key='index',
                                             value_key='value',
                                             dtype=tf.float32, size=10)}

                We assume that sparse features contains an array with records
                that contain an 'index' field that MUST BE LONG and an 'value'
                field with floats (single precision).

      batch_size: Items in a batch, must be > 0

      num_parallel_calls: Number of parallel calls

      label_key: The label key, if None no label will be returned

      num_epochs: The number of epochs. If number of epochs is set to None we
                  cycle infinite times and drop the remainder automatically.
                  This will make all batch sizes the same size and static.

      input_stream_buffer_size: The size of the input stream buffer in By

      avro_data_buffer_size: The size of the avro data buffer in By

    """
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        n_filenames = tf.shape(filenames, out_type=tf.int64)[0]
        dataset = dataset.shuffle(n_filenames, shuffle_seed)

    if label_keys is None:
        label_keys = []

    # Handle the case where the user only provided a single label key
    if not isinstance(label_keys, list):
        label_keys = [label_keys]

    for label_key in label_keys:
        if label_key not in features:
            raise ValueError(
                "`label_key` provided (%r) must be in `features`." % label_key
            )

    def filename_to_dataset(filename):
        # Batches
        return _AvroDataset(
            filenames=filename,
            features=features,
            reader_schema=reader_schema,
            batch_size=batch_size,
            drop_remainder=num_epochs is None,
            num_parallel_calls=num_parallel_calls,
            input_stream_buffer_size=input_stream_buffer_size,
            avro_data_buffer_size=avro_data_buffer_size,
        )

    # Read files sequentially (if num_parallel_reads=1) or in parallel
    dataset = dataset.interleave(
        filename_to_dataset,
        cycle_length=num_parallel_calls,
        num_parallel_calls=num_parallel_reads,
    )

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size, shuffle_seed)
    if num_epochs != 1:
        dataset = dataset.repeat(num_epochs)

    if any(isinstance(feature, tf.io.SparseFeature) for _, feature in features.items()):
        # pylint: disable=protected-access
        # pylint: disable=g-long-lambda
        dataset = dataset.map(
            lambda x: construct_tensors_for_composite_features(features, x),
            num_parallel_calls=num_parallel_calls,
        )

    # Take care of sparse shape assignment in features
    def reshape_sp_function(tensor_features):
        """Note, that sparse merge produces a rank of 2*n instead of n+1 when
        merging n dimensional tensors. But the index is produced with rank n+1.
        We correct the shape here through this method.
        :param tensor_features: the output features dict from avrodataset
        """
        for feature_name, feature in features.items():
            if (
                isinstance(feature, tf.io.SparseFeature)
                and isinstance(feature.size, list)
                and len(feature.size) > 1
            ):
                # Have -1 for unknown batch
                reshape = [-1] + list(feature.size)
                tensor_features[feature_name] = tf.sparse.reshape(
                    tensor_features[feature_name], reshape
                )
        return tensor_features

    dataset = dataset.map(reshape_sp_function, num_parallel_calls=num_parallel_calls)

    if len(label_keys) > 0:
        dataset = dataset.map(
            lambda x: (x, {label_key_: x.pop(label_key_) for label_key_ in label_keys}),
            num_parallel_calls=num_parallel_calls,
        )

    dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset
