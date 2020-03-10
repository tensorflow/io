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
import collections
import re

from tensorflow.python.ops import parsing_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging

from tensorflow_io.core.python.ops import core_ops


# Adjusted from
# https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/ops/parsing_ops.py
# Note, there are several changes to 2.1.0
# Only copied parts from `parse_example_v2` and `_parse_example_raw`
def parse_avro(serialized, reader_schema, features, avro_names=None, name=None):
    if not features:
        raise ValueError("Missing: features was %s." % features)
    features = _prepend_none_dimension(_build_keys_for_sparse_features(features))
    (sparse_keys, sparse_types, dense_keys, dense_types, dense_defaults,
     dense_shapes) = _features_to_raw_params(
        features,
        [parsing_ops.VarLenFeature, parsing_ops.SparseFeature, parsing_ops.FixedLenFeature])

    #reader_schema_tensor = ops.convert_to_tensor(reader_schema, dtypes.string, name="reader_schema")

    outputs = _parse_avro(
        serialized, reader_schema, avro_names, sparse_keys, sparse_types, dense_keys,
        dense_types, dense_defaults, dense_shapes, name)
    return parsing_ops._construct_sparse_tensors_for_sparse_features(features, outputs)


def _prepend_none_dimension(features):
    if features:
        modified_features = dict(features)  # Create a copy to modify
        for key, feature in features.items():
            if isinstance(feature, parsing_ops.FixedLenFeature):
                modified_features[key] = parsing_ops.FixedLenFeature(
                    [None] + list(feature.shape),
                    feature.dtype,
                    feature.default_value)
        return modified_features
    else:
        return features


def _parse_avro(serialized,
                reader_schema,
                names=None,
                sparse_keys=None,
                sparse_types=None,
                dense_keys=None,
                dense_types=None,
                dense_defaults=None,
                dense_shapes=None,
                name=None):
    """Parses Avro records.
    Args:
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos.
    reader_schema: A scalar string representing the reader schema.
    names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos.
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
    dense_defaults: A dict mapping string keys to `Tensor`s.
      The keys of the dict must match the dense_keys of the feature.
    dense_shapes: A list of tuples with the same length as `dense_keys`.
      The shape of the data for each dense feature referenced by `dense_keys`.
      Required for any input tensors identified by `dense_keys`.  Must be
      either fully defined, or may contain an unknown first dimension.
      An unknown first dimension means the feature is treated as having
      a variable number of blocks, and the output shape along this dimension
      is considered unknown at graph build time.  Padding is applied for
      minibatch elements smaller than the maximum number of blocks for the
      given feature along this dimension.
    name: A name for this operation (optional).
    Returns:
      A `dict` mapping keys to `Tensor`s and `SparseTensor`s.
    """
    with ops.name_scope(name, "ParseAvro", [serialized, names]):
        (names, dense_defaults_vec, sparse_keys, sparse_types,
         dense_keys, dense_shapes, _) = _process_raw_parameters(
            names, dense_defaults, sparse_keys, sparse_types, dense_keys,
            dense_types, dense_shapes)

        outputs = core_ops.io_parse_avro(
            serialized=serialized,
            reader_schema=reader_schema,
            names=names,
            dense_defaults=dense_defaults_vec,
            sparse_keys=sparse_keys,
            sparse_types=sparse_types,
            num_sparse=len(sparse_keys),
            dense_keys=dense_keys,
            dense_shapes=dense_shapes,
            name=name)

        (sparse_indices, sparse_values, sparse_shapes, dense_values) = outputs

        sparse_tensors = [
            sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape)
            in zip(sparse_indices, sparse_values, sparse_shapes)]

        return dict(zip(sparse_keys + dense_keys, sparse_tensors + dense_values))


def _build_keys_for_sparse_features(features):
    """
    Builds the fully qualified names for keys of sparse features.

    :param features:  A map of features with keys to TensorFlow features.

    :return: A map of features where for the sparse feature the 'index_key' and the 'value_key' have been expanded
             properly for the parser in the native code.
    """
    def resolve_key(parser_key, index_or_value_key):
        if not index_or_value_key.startswith('@'):
            return parser_key + '[*].' + index_or_value_key
        else:
            return index_or_value_key[1:]

    def resolve_index_key(key_, index_key):
        if isinstance(index_key, list):
            return [resolve_key(key_, index_key_) for index_key_ in index_key]
        else:
            return resolve_key(key_, index_key)

    if features:
        modified_features = dict(features)  # Create a copy to modify
        # NOTE: We iterate over sorted keys to keep things deterministic.
        for key, feature in features.items():
            if isinstance(feature, parsing_ops.SparseFeature):
                modified_features[key] = parsing_ops.SparseFeature(
                    index_key=resolve_index_key(key, feature.index_key),
                    value_key=resolve_key(key, feature.value_key),
                    dtype=feature.dtype,
                    size=feature.size,
                    already_sorted=feature.already_sorted)
        return modified_features
    else:
        return features


# Pulled this method from tensorflow/python/ops/parsing_ops.py
# here to get the same behavior but for FixedLenFeature instead of FixedLenSequenceFeature
def _prepend_none_dimension(features):
    """Returns a copy of features with adjusted FixedLenSequenceFeature shapes."""
    if features:
        modified_features = dict(features)  # Create a copy to modify
        for key, feature in features.items():
            if isinstance(feature, parsing_ops.FixedLenFeature):
                modified_features[key] = parsing_ops.FixedLenFeature(
                    [None] + list(feature.shape),
                    feature.dtype,
                    feature.default_value)
        return modified_features
    else:
        return features


# Pulled this method from tensorflow/python/ops/parsing_ops.py
# changed the following
# - removed FixedLenSequenceFeature
# - removed checks about None dimension in FixedLenFeature
# -- since it acts as FixedLenSequenceFeature, there is no need for both concepts
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
            if isinstance(feature, parsing_ops.VarLenFeature):
                if parsing_ops.VarLenFeature not in types:
                    raise ValueError("Unsupported VarLenFeature %s." % (feature,))
                if not feature.dtype:
                    raise ValueError("Missing type for feature %s." % key)
                sparse_keys.append(key)
                sparse_types.append(feature.dtype)
            elif isinstance(feature, parsing_ops.SparseFeature):
                if parsing_ops.SparseFeature not in types:
                    raise ValueError("Unsupported SparseFeature %s." % (feature,))

                if not feature.index_key:
                    raise ValueError(
                        "Missing index_key for SparseFeature %s." % (feature,))
                if not feature.value_key:
                    raise ValueError(
                        "Missing value_key for SparseFeature %s." % (feature,))
                if not feature.dtype:
                    raise ValueError("Missing type for feature %s." % key)
                index_keys = feature.index_key
                if isinstance(index_keys, str):
                    index_keys = [index_keys]
                elif len(index_keys) > 1:
                    tf_logging.warning("SparseFeature is a complicated feature config "
                                       "and should only be used after careful "
                                       "consideration of VarLenFeature.")
                for index_key in sorted(index_keys):
                    if index_key in sparse_keys:
                        dtype = sparse_types[sparse_keys.index(index_key)]
                        if dtype != dtypes.int64:
                            raise ValueError("Conflicting type %s vs int64 for feature %s." %
                                             (dtype, index_key))
                    else:
                        sparse_keys.append(index_key)
                        sparse_types.append(dtypes.int64)
                if feature.value_key in sparse_keys:
                    dtype = sparse_types[sparse_keys.index(feature.value_key)]
                    if dtype != feature.dtype:
                        raise ValueError("Conflicting type %s vs %s for feature %s." % (
                            dtype, feature.dtype, feature.value_key))
                else:
                    sparse_keys.append(feature.value_key)
                    sparse_types.append(feature.dtype)
            elif isinstance(feature, parsing_ops.FixedLenFeature):
                if parsing_ops.FixedLenFeature not in types:
                    raise ValueError("Unsupported FixedLenFeature %s." % (feature,))
                if not feature.dtype:
                    raise ValueError("Missing type for feature %s." % key)
                if feature.shape is None:
                    raise ValueError("Missing shape for feature %s." % key)
                dense_keys.append(key)
                dense_shapes.append(feature.shape)
                dense_types.append(feature.dtype)
                if feature.default_value is not None:
                    dense_defaults[key] = feature.default_value
            else:
                raise ValueError("Invalid feature %s:%s." % (key, feature))
    return (
        sparse_keys, sparse_types, dense_keys, dense_types, dense_defaults,
        dense_shapes)


# Pulled this method from tensorflow/python/ops/parsing_ops.py
# here to customize the handling of default values because
# we have
# - more types
# - handling had to change because we don't have a batch dimension when
#   calling this method
def _process_raw_parameters(names, dense_defaults, sparse_keys, sparse_types,
                            dense_keys, dense_types, dense_shapes):
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
    dense_defaults = collections.OrderedDict(
    ) if dense_defaults is None else dense_defaults
    sparse_keys = [] if sparse_keys is None else sparse_keys
    sparse_types = [] if sparse_types is None else sparse_types
    dense_keys = [] if dense_keys is None else dense_keys
    dense_types = [] if dense_types is None else dense_types
    dense_shapes = ([[]] * len(dense_keys)
                    if dense_shapes is None else dense_shapes)

    num_dense = len(dense_keys)
    num_sparse = len(sparse_keys)

    if len(dense_shapes) != num_dense:
        raise ValueError("len(dense_shapes) != len(dense_keys): %d vs. %d" %
                         (len(dense_shapes), num_dense))
    if len(dense_types) != num_dense:
        raise ValueError("len(dense_types) != len(num_dense): %d vs. %d" %
                         (len(dense_types), num_dense))
    if len(sparse_types) != num_sparse:
        raise ValueError("len(sparse_types) != len(sparse_keys): %d vs. %d" %
                         (len(sparse_types), num_sparse))
    if num_dense + num_sparse == 0:
        raise ValueError("Must provide at least one sparse key or dense key")
    if not set(dense_keys).isdisjoint(set(sparse_keys)):
        raise ValueError(
            "Dense and sparse keys must not intersect; intersection: %s" %
            set(dense_keys).intersection(set(sparse_keys)))

    # Convert dense_shapes to TensorShape object.
    dense_shapes = [tensor_shape.as_shape(shape) for shape in dense_shapes]

    dense_defaults_vec = []
    for i, key in enumerate(dense_keys):
        default_value = dense_defaults.get(key)
        dense_shape = dense_shapes[i]

        # Whenever the user did not provide a default, set it

        # ************* START difference: This part is different from the originally copied code ***************
        if default_value is None:
            if dense_types[i] == dtypes.string:
                default_value = ""
            elif dense_types[i] == dtypes.bool:
                default_value = False
            else:  # Should be numeric type
                default_value = 0
            default_value = ops.convert_to_tensor(
                default_value, dtype=dense_types[i])
        elif not isinstance(default_value, ops.Tensor):
            key_name = "key_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
            default_value = ops.convert_to_tensor(
                default_value, dtype=dense_types[i], name=key_name)
            # If we have a shape and the first dimension is not None
            if dense_shape.rank and dense_shape.dims[0].value:
                default_value = array_ops.reshape(default_value, dense_shape)
        # ************* END difference: This part is different from the originally copied code *****************
        dense_defaults_vec.append(default_value)

    # Finally, convert dense_shapes to TensorShapeProto
    dense_shapes_as_proto = [shape.as_proto() for shape in dense_shapes]

    return (names, dense_defaults_vec, sparse_keys, sparse_types, dense_keys,
            dense_shapes_as_proto, dense_shapes)
