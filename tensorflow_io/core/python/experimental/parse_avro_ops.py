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
"""parse_avro_ops"""

import collections
import re

import tensorflow as tf
import tensorflow_io
from tensorflow_io.core.python.ops import core_ops

# Adjusted from
# https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/ops/parsing_ops.py
# Note, there are several changes to 2.1.0
# Only copied parts from `parse_example_v2` and `_parse_example_raw`


def parse_avro(serialized, reader_schema, features, avro_names=None, name=None):
    """
    Parses `avro` records into a `dict` of tensors.

    This op parses serialized avro records into a dictionary mapping keys to
    `Tensor`, and `SparseTensor` objects. `features` is a dict from keys to
    `VarLenFeature`, `SparseFeature`, `RaggedFeature`, and `FixedLenFeature`
    objects. Each `VarLenFeature` and `SparseFeature` is mapped to a
    `SparseTensor`; each `FixedLenFeature` is mapped to a `Tensor`.

    Each `VarLenFeature` maps to a `SparseTensor` of the specified type
    representing a ragged matrix. Its indices are `[batch, index]` where `batch`
    identifies the example in `serialized`, and `index` is the value's index in
    the list of values associated with that feature and example.

    Each `SparseFeature` maps to a `SparseTensor` of the specified type
    representing a Tensor of `dense_shape` `[batch_size] + SparseFeature.size`.
    Its `values` come from the feature in the examples with key `value_key`.
    A `values[i]` comes from a position `k` in the feature of an example at batch
    entry `batch`. This positional information is recorded in `indices[i]` as
    `[batch, index_0, index_1, ...]` where `index_j` is the `k-th` value of
    the feature in the example at with key `SparseFeature.index_key[j]`.
    In other words, we split the indices (except the first index indicating the
    batch entry) of a `SparseTensor` by dimension into different features of the
    avro record. Due to its complexity a `VarLenFeature` should be preferred
    over a `SparseFeature` whenever possible.

    Each `FixedLenFeature` `df` maps to a `Tensor` of the specified type (or
    `tf.float32` if not specified) and shape `(serialized.size(),) + df.shape`.
    `FixedLenFeature` entries with a `default_value` are optional. With no default
    value, we will fail if that `Feature` is missing from any example in
    `serialized`.

    Use this within the dataset.map(parser_fn=parse_avro).

    Only works for batched serialized input!

    Args:
        serialized: The batched, serialized string tensors.

        reader_schema: The reader schema. Note, this MUST match the reader schema
        from the avro_record_dataset. Otherwise, this op will segfault!

        features: A map of feature names mapped to feature information.

        avro_names: (Optional.) may contain descriptive names for the
        corresponding serialized avro parts. These may be useful for debugging
        purposes, but they have no effect on the output. If not `None`,
        `avro_names` must be the same length as `serialized`.

        name: The name of the op.

    Returns:
        A map of feature names to tensors.
    """
    if not features:
        raise ValueError("Missing: features was %s." % features)
    features = _build_keys_for_sparse_features(features)
    (
        sparse_keys,
        sparse_types,
        sparse_ranks,
        dense_keys,
        dense_types,
        dense_defaults,
        dense_shapes,
    ) = _features_to_raw_params(
        features,
        [
            tensorflow_io.experimental.columnar.VarLenFeatureWithRank,
            tf.io.SparseFeature,
            tf.io.FixedLenFeature,
        ],
    )

    outputs = _parse_avro(
        serialized,
        reader_schema,
        avro_names,
        sparse_keys,
        sparse_types,
        sparse_ranks,
        dense_keys,
        dense_types,
        dense_defaults,
        dense_shapes,
        name,
    )
    return construct_tensors_for_composite_features(features, outputs)


def _parse_avro(
    serialized,
    reader_schema,
    names=None,
    sparse_keys=None,
    sparse_types=None,
    sparse_ranks=None,
    dense_keys=None,
    dense_types=None,
    dense_defaults=None,
    dense_shapes=None,
    name=None,
    avro_num_minibatches=0,
):
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
        sparse_ranks: ranks of sparse feature. `tf.int64` (`Int64List`) is supported.
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
    with tf.name_scope(name or "ParseAvro"):
        (
            names,
            dense_defaults_vec,
            sparse_keys,
            sparse_types,
            dense_keys,
            dense_shapes,
            _,
        ) = _process_raw_parameters(
            names,
            dense_defaults,
            sparse_keys,
            sparse_types,
            dense_keys,
            dense_types,
            dense_shapes,
        )

        outputs = core_ops.io_parse_avro(
            serialized=serialized,
            reader_schema=reader_schema,
            names=names,
            dense_defaults=dense_defaults_vec,
            sparse_keys=sparse_keys,
            sparse_types=sparse_types,
            num_sparse=len(sparse_keys),
            sparse_ranks=sparse_ranks,
            dense_keys=dense_keys,
            dense_shapes=dense_shapes,
            name=name,
            avro_num_minibatches=avro_num_minibatches,
        )

        (sparse_indices, sparse_values, sparse_shapes, dense_values) = outputs

        sparse_tensors = [
            tf.sparse.SparseTensor(ix, val, shape)
            for (ix, val, shape) in zip(sparse_indices, sparse_values, sparse_shapes)
        ]

        return dict(zip(sparse_keys + dense_keys, sparse_tensors + dense_values))


# Adjusted from
# https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/ops/parsing_ops.py
# _prepend_none_dimension with the following changes
# - Removed the warning
# - Switched this to FixedLenFeature -- instead of FixedLenSequenceFeature
def _prepend_none_dimension(features):
    """prepend_none_dimension"""
    if features:
        modified_features = dict(features)  # Create a copy to modify
        for key, feature in features.items():
            if isinstance(feature, tf.io.FixedLenFeature):
                modified_features[key] = tf.io.FixedLenFeature(
                    [None] + list(feature.shape), feature.dtype, feature.default_value
                )
        return modified_features
    return features


def _build_keys_for_sparse_features(features):
    """
    Builds the fully qualified names for keys of sparse features.

    Args:
        features:  A map of features with keys to TensorFlow features.

    Returns:
        A map of features where for the sparse feature
        the 'index_key' and the 'value_key' have been expanded
        properly for the parser in the native code.
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
        modified_features = dict(features)  # Create a copy to modify
        # NOTE: We iterate over sorted keys to keep things deterministic.
        for key, feature in features.items():
            if isinstance(feature, tf.io.SparseFeature):
                modified_features[key] = tf.io.SparseFeature(
                    index_key=resolve_index_key(key, feature.index_key),
                    value_key=resolve_key(key, feature.value_key),
                    dtype=feature.dtype,
                    size=feature.size,
                    already_sorted=feature.already_sorted,
                )
        return modified_features
    return features


# adapted from https://github.com/tensorflow/tensorflow/blob/6d0f422525d8c1dd3184d39494abacd32b52a840/tensorflow/python/ops/parsing_config.py#L661 and skipped RaggedFeature part
def construct_tensors_for_composite_features(features, tensor_dict):
    """construct_tensors_for_composite_features"""
    tensor_dict = dict(tensor_dict)  # Do not modify argument passed in.
    updates = {}
    for key in sorted(features.keys()):
        feature = features[key]
        if isinstance(feature, tf.io.SparseFeature):
            # Construct SparseTensors for SparseFeatures
            if isinstance(feature.index_key, str):
                sp_ids = tensor_dict[feature.index_key]
            else:
                sp_ids = [tensor_dict[index_key] for index_key in feature.index_key]
            sp_values = tensor_dict[feature.value_key]
            updates[key] = tf.compat.v1.sparse_merge(
                sp_ids,
                sp_values,
                vocab_size=feature.size,
                already_sorted=feature.already_sorted,
            )

    # Process updates after all composite tensors have been constructed (in case
    # multiple features use the same value_key, and one uses that key as its
    # feature key).
    tensor_dict.update(updates)

    # Remove tensors from dictionary that were only used to construct
    # tensors for SparseFeature or RaggedTensor.
    for key in set(tensor_dict) - set(features):
        del tensor_dict[key]
    return tensor_dict


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
    sparse_ranks = []
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
            if isinstance(
                feature, tensorflow_io.experimental.columnar.VarLenFeatureWithRank
            ):
                _handle_varlen_feature(
                    feature, key, sparse_keys, sparse_types, sparse_ranks, types
                )
            elif isinstance(feature, tf.io.SparseFeature):
                _handle_sparse_feature(
                    feature, key, sparse_keys, sparse_types, sparse_ranks, types
                )
            elif isinstance(feature, tf.io.FixedLenFeature):
                _handle_fixedlen_feature(
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
        sparse_ranks,
        dense_keys,
        dense_types,
        dense_defaults,
        dense_shapes,
    )


def _handle_fixedlen_feature(
    dense_defaults, dense_keys, dense_shapes, dense_types, feature, key, types
):
    """handle_fixedlen_feature"""
    if tf.io.FixedLenFeature not in types:
        raise ValueError("Unsupported FixedLenFeature {}.".format(feature))
    if not feature.dtype:
        raise ValueError("Missing type for feature %s." % key)
    if feature.shape is None:
        raise ValueError("Missing shape for feature %s." % key)
    dense_keys.append(key)
    dense_shapes.append(feature.shape)
    dense_types.append(feature.dtype)
    if feature.default_value is not None:
        dense_defaults[key] = feature.default_value


def _handle_sparse_feature(
    feature, key, sparse_keys, sparse_types, sparse_ranks, types
):
    """handle_sparse_feature"""
    if tf.io.SparseFeature not in types:
        raise ValueError("Unsupported SparseFeature {}.".format(feature))
    if not feature.index_key:
        raise ValueError("Missing index_key for SparseFeature {}.".format(feature))
    if not feature.value_key:
        raise ValueError("Missing value_key for SparseFeature {}.".format(feature))
    if not feature.dtype:
        raise ValueError("Missing type for feature %s." % key)
    index_keys = feature.index_key
    if isinstance(index_keys, str):
        index_keys = [index_keys]
    elif len(index_keys) > 1:
        tf.get_logger().warning(
            "SparseFeature is a complicated feature config "
            "and should only be used after careful "
            "consideration of VarLenFeature."
        )
    for index_key in sorted(index_keys):
        if index_key in sparse_keys:
            dtype = sparse_types[sparse_keys.index(index_key)]
            if dtype != tf.int64:
                raise ValueError(
                    "Conflicting type {} vs int64 for feature {}.".format(
                        dtype, index_key
                    )
                )
        else:
            sparse_keys.append(index_key)
            sparse_types.append(tf.int64)
            # sparse features always have rank 1 because they encode the indices separately (one for each component) and then merge these before the user get's them.
            # setting 1 here is merely achieving the same behavior as before.
            sparse_ranks.append(1)
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
        # sparse features always have rank 1 because they encode the indices separately (one for each component) and then merge these before the user get's them.
        # setting 1 here is merely achieving the same behavior as before.
        sparse_ranks.append(1)


def _handle_varlen_feature(
    feature, key, sparse_keys, sparse_types, sparse_ranks, types
):
    """handle_varlen_feature"""
    if tensorflow_io.experimental.columnar.VarLenFeatureWithRank not in types:
        raise ValueError("Unsupported VarLenFeatureWithRank {}.".format(feature))
    if not feature.dtype:
        raise ValueError("Missing type for VarLenFeatureWithRank %s." % key)
    if not feature.rank:
        raise ValueError("Missing rank for VarLenFeatureWithRank %s." % key)
    sparse_keys.append(key)
    sparse_types.append(feature.dtype)
    sparse_ranks.append(feature.rank)


# Pulled this method from tensorflow/python/ops/parsing_ops.py
# here to customize the handling of default values because
# we have
# - more types
# - handling had to change because we don't have a batch dimension when
#   calling this method
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
            default_value = tf.constant([], dtype=dense_types[i])
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
