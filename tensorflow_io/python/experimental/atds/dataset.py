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
"""ATDSDataset"""

from typing import Optional

import tensorflow as tf
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import nest
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util

from tensorflow_io.python.ops import core_ops
from tensorflow_io.python.experimental.atds.features import (
    DenseFeature,
    SparseFeature,
    VarlenFeature,
)

# Argument default values used in ATDS Dataset.
_DEFAULT_DROP_REMAINDER = False  # Do not drop last batch.
_DEFAULT_READER_BUFFER_SIZE_BYTES = 128 * 1024  # 128 KB
_DEFAULT_SHUFFLE_BUFFER_SIZE_EXAMPLES = 0  # shuffle is disabled.
_DEFAULT_NUM_PARALLEL_CALLS = 1  # process sequentially.

# Feature type name used in ATDS Dataset Op.
_DENSE_FEATURE_TYPE = "dense"
_SPARSE_FEATURE_TYPE = "sparse"
_VARLEN_FEATURE_TYPE = "varlen"

# Supported feature configs
_SUPPORTED_FEATURE_CONFIG = (DenseFeature, SparseFeature, VarlenFeature)


class ATDSDataset(dataset_ops.DatasetSource):
    """A `Dataset` comprising records from one or more Avro files.

    This dataset load Avro records from the files into a dict of tensors.
    The output dict has feature name as key and tf.Tensor or tf.SparseTensor
    as value. The output tensor values are batched with the user defined
    batch size.

    Shuffle can be enabled before batch by configuring shuffle buffer size.
    The shuffle buffer size dictates the elements *in addition* to the batch size
    that would be read and sampled.
    This dataset keeps collecting Avro blocks(a sequence of Avro records),
    until the total number of unread records is greater than the shuffle
    buffer size + batch_size, then randomly samples block from the collected blocks.
    An Avro Record from the sampled block will be parsed and batched into
    the output tensors.

    For instance, assume your dataset contains 5 blocks with 100 records in
    each block. When the batch size is set to 32 and shuffle buffer size is set
    to 128, this dataset will collect two blocks as the two blocks contains more
    than 128 + 32 = 160 unread records, and randomly samples block from the two
    blocks 32 times.
    When a block is sampled, a record in the sampled block is read and batched
    into the output tensor dict until all records in the sampled block are read.
    If only one block fits into the batch + shuffle_buffer_size, records in that
    block will be read sequentially without shuffle. Users can increase the
    shuffle buffer size or apply dataset unbatch, shuffle, and batch for better shuffling.

    The memory footprint of this shuffle buffer is signficantly different from tf.data.Dataset.shuffle
    In Tensorflow's shuffle, the shuffle buffer specifies a separate buffer of elements to pick
    random elements from. In this implementation, the shuffle buffer + batch forms the total number of
    elements that would be read for sampling.

    Here's an example comparing shuffle between AvroReader and Tensorflow:

    Data size is 1000
    Batch size is 64

    Case 1: perfect shuffle
    Shuffle buffer size 1000
    TF: shuffle elements(1, 1000) to create a batch of size 64
    AvroReader: shuffle elements(1, 1000) to create a batch of size 64

    Case 2: not perfect but shuffle > batch
    Shuffle buffer size is 256
    TF:
    Shuffle elements(1, 256) to pick 1 element
    Shuffle elements(1, 257) to pick 1 element
    …
    shuffle(1, 320) to create a batch of size 64

    AvroReader:
    Shuffle elements(1, 320) to create a batch of size 64

    Case 3: shuffle buffer < batch
    Shuffle buffer size is 32
    TF:
    Shuffle elements(1, 32) to pick 1 element
    Shuffle elements(1, 33) to pick 1 element
    …
    shuffle(1, 96) to create a batch of size 64

    ATDS: Shuffle elements(1, 96) to create a batch of size 64

    Case 4: no shuffle
    Shuffle buffer size is 0
    Tensorflow and ATDS both will just directly read to create a batch of size 64


    A minimal example is given below:

    >>> import tempfile
    >>> import avro.schema
    >>> from avro.datafile import DataFileWriter
    >>> from avro.io import DatumWriter
    >>> from tensorflow_io.python.experimental.atds.dataset import ATDSDataset
    >>> from tensorflow_io.python.experimental.atds.features import DenseFeature
    >>> example_path = os.path.join(tempfile.gettempdir(), "example.avro")
    >>> np.random.seed(0)

    >>> # Define Avro schema in ATDS format.
    >>> json_schema = '''{
    ...     "type": "record",
    ...     "name": "example",
    ...     "fields": [
    ...         { "name": "x", "type": "float" },
    ...         { "name": "y", "type": "float" }
    ...     ]
    ... }'''
    >>> schema = avro.schema.Parse(json_schema)

    >>> # Write the Avro records to a file.
    >>> with open(example_path, "wb") as f:
    ...     writer = DataFileWriter(f, DatumWriter(), schema)
    ...     for _ in range(3):
    ...         x, y = np.random.random(), np.random.random()
    ...         writer.append({"x": x, "y": y})
    ...     writer.close()

    >>> # Read the data back out.
    >>> feature_config = {
    ...     "x": DenseFeature([], dtype=tf.float32),
    ...     "y": DenseFeature([], dtype=tf.float32)
    ... }
    >>> for batch in ATDSDataset([example_path], batch_size=2,
    ...                         features=feature_config):
    ...     print("x = {x},  y = {y}".format(**batch))
    x = [0.5488135  0.60276335],  y = [0.71518934 0.5448832 ]
    x = [0.4236548],  y = [0.6458941]
    """

    def __init__(
        self,
        filenames,
        batch_size,
        features,
        drop_remainder=False,
        reader_buffer_size=None,
        shuffle_buffer_size=None,
        num_parallel_calls=None,
    ):
        """Creates a `ATDSDataset` to read one or more Avro files encoded with
           ATDS Schema.

           Each element of the dataset contains an Avro Record that will be
           parsed into a dict of tensors.

        Args:
          filenames: A `tf.string` tensor containing one or more filenames.
          batch_size: A `tf.int64` scalar representing the number of records to
            read and parse per iteration.
          features: A feature configuration dict with feature name as key and
            ATDS feature as value. ATDS features can be one of the DenseFeature,
            SparseFeature, or VarlenFeature. See
            tensorflow_io.python.experimental.atds.features for more details.
          drop_remainder: (Optional.) A `tf.bool` scalar tf.Tensor, representing
            whether the last batch should be dropped in the case it has fewer
            than batch_size elements. The default behavior is not to drop the
            smaller batch.
          reader_buffer_size: (Optional.) A `tf.int64` scalar representing the
            number of bytes used in the file content buffering.
          shuffle_buffer_size: (Optional.) A `tf.int64` scalar representing the
            number of records to shuffle together before batching. If not
            specified, data is batched without shuffle.
          num_parallel_calls: (Optional.) A `tf.int64` scalar representing the
            maximum thread number used in the dataset. If greater than one,
            records in files are processed in parallel with deterministic order.
            The number will be truncated when it is greater than the maximum
            available parallelism number on the host. If set to `tf.data.AUTOTUNE`,
            number of threads will be adjusted dynamically based on workload and
            available resources. If not specified, records will be processed sequentially.

        Raises:
          TypeError: If any argument does not have the expected type.
          ValueError: If any argument does not have the expected shape
                      or features have invalid config.
        """
        self._filenames = filenames
        self._batch_size = batch_size
        self._drop_remainder = convert.optional_param_to_tensor(
            "drop_remainder",
            drop_remainder,
            argument_default=_DEFAULT_DROP_REMAINDER,
            argument_dtype=tf.bool,
        )
        self._reader_buffer_size = convert.optional_param_to_tensor(
            "reader_buffer_size",
            reader_buffer_size,
            argument_default=_DEFAULT_READER_BUFFER_SIZE_BYTES,
        )
        self._shuffle_buffer_size = convert.optional_param_to_tensor(
            "shuffle_buffer_size",
            shuffle_buffer_size,
            argument_default=_DEFAULT_SHUFFLE_BUFFER_SIZE_EXAMPLES,
        )
        self._num_parallel_calls = convert.optional_param_to_tensor(
            "num_parallel_calls",
            num_parallel_calls,
            argument_default=_DEFAULT_NUM_PARALLEL_CALLS,
        )

        if features is None or not isinstance(features, dict):
            raise ValueError(
                f"Features can only be a dict with feature name as key"
                f" and ATDS feature configuration as value but found {features}."
                f" Available feature configuration are {_SUPPORTED_FEATURE_CONFIG}."
            )
        if not features:
            raise ValueError(
                "Features dict cannot be empty and should have at " "least one feature."
            )

        feature_keys = []
        feature_types = []
        sparse_dtypes = []
        sparse_shapes = []

        element_spec = {}
        for key in sorted(features):
            feature = features[key]
            if not isinstance(feature, _SUPPORTED_FEATURE_CONFIG):
                raise ValueError(
                    f"Unknown ATDS feature configuration {feature}. "
                    f"Only {_SUPPORTED_FEATURE_CONFIG} are supported."
                )

            feature_keys.append(key)
            shape = [dim if dim != -1 else None for dim in feature.shape]
            if isinstance(feature, DenseFeature):
                feature_types.append(_DENSE_FEATURE_TYPE)
                element_spec[key] = tf.TensorSpec(shape, feature.dtype)
            elif isinstance(feature, SparseFeature):
                feature_types.append(_SPARSE_FEATURE_TYPE)
                sparse_dtypes.append(feature.dtype)
                sparse_shapes.append(shape)
                element_spec[key] = tf.SparseTensorSpec(shape, feature.dtype)
            elif isinstance(feature, VarlenFeature):
                feature_types.append(_VARLEN_FEATURE_TYPE)
                sparse_dtypes.append(feature.dtype)
                sparse_shapes.append(shape)
                element_spec[key] = tf.SparseTensorSpec(shape, feature.dtype)

        constant_drop_remainder = tensor_util.constant_value(self._drop_remainder)
        if constant_drop_remainder:
            constant_batch_size = tensor_util.constant_value(self._batch_size)
            self._element_spec = nest.map_structure(
                lambda spec: spec._batch(constant_batch_size), element_spec
            )
        else:
            self._element_spec = nest.map_structure(
                lambda spec: spec._batch(None), element_spec
            )

        variant_tensor = core_ops.io_atds_dataset(
            filenames=self._filenames,
            batch_size=self._batch_size,
            drop_remainder=self._drop_remainder,
            reader_buffer_size=self._reader_buffer_size,
            shuffle_buffer_size=self._shuffle_buffer_size,
            num_parallel_calls=self._num_parallel_calls,
            feature_keys=feature_keys,
            feature_types=feature_types,
            sparse_dtypes=sparse_dtypes,
            sparse_shapes=sparse_shapes,
            output_dtypes=structure.get_flat_tensor_types(self._element_spec),
            output_shapes=structure.get_flat_tensor_shapes(self._element_spec),
        )
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec
