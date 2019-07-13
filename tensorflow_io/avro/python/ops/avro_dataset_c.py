# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/data/experimental/ops/readers.py

# Parse example dataset
# https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/python/data/experimental/ops/parsing_ops.py

# For tf export use examples see
# sql_dataset_test_base.py for use and readers.py for definition

import collections
import re

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.data.util import structure
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.dataset_ops import DatasetSource, DatasetV1Adapter
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.experimental.ops import readers
from tensorflow_io.core.python.ops import core_ops as avro_ops
# from tensorflow.python.util.tf_export import tf_export


# Note: I've hidden the dataset because it does not apply the mapping for
# sparse tensors
# Such mapping is not possible inside the dataset, rather it needs to happen
# through a map on the output of the dataset which is a map of keys to tensors
# This can be changed when eager mode is the default and the only mode supported
class _AvroDataset(DatasetSource):
  """A `DatasetSource` that reads and parses Avro records from files."""

  def __init__(self, filenames, features, reader_schema="", batch_size=128,
      drop_remainder=False, num_parallel_calls=2):

    super(_AvroDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(
        filenames, dtypes.string, name="filenames")
    self._features = _AvroDataset._build_keys_for_sparse_features(features)
    self._reader_schema = reader_schema
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder
    self._num_parallel_calls = num_parallel_calls

    # Copied from _ParseExampleDataset from data/experimental/ops/parsing_ops.py
    (sparse_keys, sparse_types, dense_keys, dense_types, dense_defaults,
     dense_shapes) = parsing_ops._features_to_raw_params(
        self._features, [
          parsing_ops.VarLenFeature, parsing_ops.SparseFeature,
          parsing_ops.FixedLenFeature
        ])

    (_, dense_defaults_vec, sparse_keys, sparse_types, dense_keys, dense_shapes,
     dense_shape_as_shape) = _AvroDataset._process_raw_parameters(
        None, dense_defaults, sparse_keys, sparse_types, dense_keys,
        dense_types, dense_shapes)

    self._sparse_keys = sparse_keys
    self._sparse_types = sparse_types
    self._dense_keys = dense_keys
    self._dense_defaults = dense_defaults_vec
    self._dense_shapes = dense_shapes
    self._dense_types = dense_types

    # Pre-pend batch dimension -- which is unknown here
    dense_output_shapes = [
      tensor_shape.TensorShape(None).concatenate(shape)
      for shape in dense_shape_as_shape
    ]
    sparse_output_shapes = [
      [None, None]  # One for batch and one for length of sparse tensor
      for _ in range(len(sparse_keys))
    ]

    output_shapes = dict(
        zip(self._dense_keys + self._sparse_keys,
            dense_output_shapes + sparse_output_shapes))
    output_types = dict(
        zip(self._dense_keys + self._sparse_keys,
            self._dense_types + self._sparse_types))
    output_classes = dict(
        zip(self._dense_keys + self._sparse_keys,
            [ops.Tensor for _ in range(len(self._dense_defaults))] +
            [sparse_tensor.SparseTensor for _ in range(len(self._sparse_keys))
             ]))
    self._structure = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)

  def _as_variant_tensor(self):
    return avro_ops.avro_dataset_c(
        filenames=self._filenames,
        batch_size=self._batch_size,
        drop_remainder=self._drop_remainder,
        reader_schema=self._reader_schema,
        dense_defaults=self._dense_defaults,
        sparse_keys=self._sparse_keys,
        dense_keys=self._dense_keys,
        sparse_types=self._sparse_types,
        dense_shapes=self._dense_shapes,
        **dataset_ops.flat_structure(self))

  @property
  def _element_structure(self):
    return self._structure

  @staticmethod
  def _build_keys_for_sparse_features(features):
    """
    Builds the fully qualified names for keys of sparse features.

    :param features:  A map of features with keys to TensorFlow features.

    :return: A map of features where for the sparse feature the 'index_key' and the 'value_key' have been expanded
             properly for the parser in the native code.
    """
    def resolve_index_key(key_, index_key):
      if isinstance(index_key, list):
        return [key_ + '[*].' + index_key_ for index_key_ in index_key]
      else:
        return key_ + '[*].' + index_key

    if features:
      # NOTE: We iterate over sorted keys to keep things deterministic.
      for key in sorted(features.keys()):
        feature = features[key]
        if isinstance(feature, parsing_ops.SparseFeature):
          features[key] = parsing_ops.SparseFeature(
              index_key=resolve_index_key(key, feature.index_key),
              value_key=key + '[*].' + feature.value_key,
              dtype=feature.dtype,
              size=feature.size,
              already_sorted=feature.already_sorted)
    return features

  # Pulled this method from tensorflow/python/ops/parsing_ops.py
  # here to customize the handling of default values because
  # we have
  # - more types
  # - handling had to change because we don't have a batch dimension when
  #   calling this method
  @staticmethod
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
        default_value = array_ops.reshape(default_value, dense_shape)
      # ************* END difference: This part is different from the originally copied code *****************
      dense_defaults_vec.append(default_value)

    # Finally, convert dense_shapes to TensorShapeProto
    dense_shapes_as_proto = [shape.as_proto() for shape in dense_shapes]

    return (names, dense_defaults_vec, sparse_keys, sparse_types, dense_keys,
            dense_shapes_as_proto, dense_shapes)

# TODO(fraudies) Fix the proper export of the symbol
# @tf_export("tensorflow_io.avro.make_avro_dataset", v1=[])
def make_avro_dataset(
    file_pattern,
    features,
    batch_size,
    reader_schema="",
    num_parallel_calls=2,
    label_key=None,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=optimization.AUTOTUNE,
    num_parallel_reads=1,
    sloppy=False
):
  """Makes an avro dataset.

  Reads from avro files and parses the contents into tensors.

  Args:
    filenames: A `tf.string` tensor containing one or more filenames.
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

    reader_schema: (Optional.) A `tf.string` scalar for schema resolution.

    num_parallel_calls: Number of parallel calls

  """
  filenames = readers._get_file_names(file_pattern, False)
  dataset = dataset_ops.Dataset.from_tensor_slices(filenames)
  if shuffle:
    dataset = dataset.shuffle(len(filenames), shuffle_seed)

  if label_key is not None and label_key not in features:
    raise ValueError("`label_key` provided must be in `features`.")

  def filename_to_dataset(filename):
    # Batches
    return _AvroDataset(
        filenames=filename,
        features=features,
        reader_schema=reader_schema,
        batch_size=batch_size,
        drop_remainder=num_epochs is None,
        num_parallel_calls=num_parallel_calls
    )

  # Read files sequentially (if num_parallel_reads=1) or in parallel
  dataset = dataset.apply(
      interleave_ops.parallel_interleave(
          filename_to_dataset, cycle_length=num_parallel_reads, sloppy=sloppy))

  dataset = readers._maybe_shuffle_and_repeat(
      dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed)

  if any(
      isinstance(feature, parsing_ops.SparseFeature)
      for _, feature in features.items()
  ):
    # pylint: disable=protected-access
    # pylint: disable=g-long-lambda
    dataset = dataset.map(
        lambda x: parsing_ops._construct_sparse_tensors_for_sparse_features(
            features, x), num_parallel_calls=num_parallel_calls)

  if label_key:
    if label_key not in features:
      raise ValueError(
          "The `label_key` provided (%r) must be one of the `features` keys." %
          label_key)
    dataset = dataset.map(lambda x: (x, x.pop(label_key)))

  dataset = dataset.prefetch(prefetch_buffer_size)

  return dataset

# TODO(fraudies) Fix the proper export of the symbol
# @tf_export(v1=["tensorflow_io.avro.make_avro_dataset"])
def make_avro_dataset_v1(
    file_pattern,
    features,
    batch_size,
    reader_schema="",
    num_parallel_calls=2,
    label_key=None,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=optimization.AUTOTUNE,
    num_parallel_reads=1,
    sloppy=False
):  # pylint: disable=missing-docstring
  return dataset_ops.DatasetV1Adapter(make_avro_dataset(
      file_pattern, features, batch_size, reader_schema, num_parallel_calls,
      label_key, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed,
      prefetch_buffer_size, num_parallel_reads, sloppy))
make_avro_dataset_v1.__doc__ = make_avro_dataset.__doc__