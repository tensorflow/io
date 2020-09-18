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
"""_AvroRecordDataset"""

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

_DEFAULT_READER_BUFFER_SIZE_BYTES = 256 * 1024  # 256 KB
_DEFAULT_READER_SCHEMA = ""
# From https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/data/ops/readers.py

# copied from https://github.com/tensorflow/tensorflow/blob/
# 3095681b8649d9a828afb0a14538ace7a998504d/tensorflow/python/data/ops/readers.py#L36
def _create_or_validate_filenames_dataset(filenames):
    """create_or_validate_filenames_dataset"""
    if isinstance(filenames, tf.data.Dataset):
        if tf.compat.v1.data.get_output_types(filenames) != tf.string:
            raise TypeError(
                "`filenames` must be a `tf.data.Dataset` of `tf.string` elements."
            )
        if not tf.compat.v1.data.get_output_shapes(filenames).is_compatible_with(
            tf.TensorShape([])
        ):
            raise TypeError(
                "`filenames` must be a `tf.data.Dataset` of scalar `tf.string` "
                "elements."
            )
    else:
        filenames = tf.convert_to_tensor(filenames, dtype_hint=tf.string)
        if filenames.dtype != tf.string:
            raise TypeError(
                "`filenames` must be a `tf.Tensor` of dtype `tf.string` dtype."
                " Got {}".format(filenames.dtype)
            )
        filenames = tf.reshape(filenames, [-1], name="flat_filenames")
        filenames = tf.data.Dataset.from_tensor_slices(filenames)

    return filenames


# copied from https://github.com/tensorflow/tensorflow/blob/
# 3095681b8649d9a828afb0a14538ace7a998504d/tensorflow/python/data/ops/readers.py#L67
def _create_dataset_reader(dataset_creator, filenames, num_parallel_reads=None):
    """create_dataset_reader"""

    def read_one_file(filename):
        filename = tf.convert_to_tensor(filename, tf.string, name="filename")
        return dataset_creator(filename)

    if num_parallel_reads is None:
        return filenames.flat_map(read_one_file)
    if num_parallel_reads == tf.data.experimental.AUTOTUNE:
        return filenames.interleave(
            read_one_file, num_parallel_calls=num_parallel_reads
        )
    return filenames.interleave(
        read_one_file, cycle_length=num_parallel_reads, block_length=1
    )


class _AvroRecordDataset(tf.data.Dataset):
    """A `Dataset` comprising records from one or more AvroRecord files."""

    def __init__(self, filenames, buffer_size=None, reader_schema=None):
        """Creates a `AvroRecordDataset`.

        Args:
          filenames: A `tf.string` tensor containing one or more filenames.
          buffer_size: (Optional.) A `tf.int64` scalar representing the number of
            bytes in the read buffer. 0 means no buffering.
          reader_schema: (Optional.) A `tf.string` scalar
          representing the reader schema or None
        """
        self._filenames = filenames
        self._buffer_size = _AvroRecordDataset.__optional_param_to_tensor(
            "buffer_size",
            buffer_size,
            argument_default=_DEFAULT_READER_BUFFER_SIZE_BYTES,
        )
        self._reader_schema = _AvroRecordDataset.__optional_param_to_tensor(
            "reader_schema",
            reader_schema,
            argument_default=_DEFAULT_READER_SCHEMA,
            argument_dtype=tf.dtypes.string,
        )
        variant_tensor = core_ops.io_avro_record_dataset(
            self._filenames, self._buffer_size, self._reader_schema
        )
        super().__init__(variant_tensor)

    # Copied from https://github.com/tensorflow/tensorflow/blob/f40a875355557483aeae60ffcf757fc9626c752b
    #            /tensorflow/python/data/util/convert.py#L26-L35
    @staticmethod
    def __optional_param_to_tensor(
        argument_name,
        argument_value,
        argument_default=0,
        argument_dtype=tf.dtypes.int64,
    ):
        """optional_param_to_tensor"""
        if argument_value is not None:
            return tf.convert_to_tensor(
                argument_value, dtype=argument_dtype, name=argument_name
            )
        return tf.constant(argument_default, dtype=argument_dtype, name=argument_name)

    @property
    def element_spec(self):
        return tf.TensorSpec([], tf.dtypes.string)

    def _inputs(self):
        return []


class AvroRecordDataset(tf.data.Dataset):
    """A `Dataset` comprising records from one or more AvroRecord files."""

    def __init__(
        self, filenames, buffer_size=None, num_parallel_reads=None, reader_schema=None
    ):
        """Creates a `AvroRecordDataset` to read one or more AvroRecord files.

        Args:
          filenames: A `tf.string` tensor or `tf.data.Dataset` containing one or
            more filenames.
          buffer_size: (Optional.) A `tf.int64` scalar representing the number of
            bytes in the read buffer. If your input pipeline is I/O bottlenecked,
            consider setting this parameter to a value 1-100 MBs. If `None`, a
            sensible default for both local and remote file systems is used.
          num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
            number of files to read in parallel. If greater than one, the records of
            files read in parallel are outputted in an interleaved order. If your
            input pipeline is I/O bottlenecked, consider setting this parameter to a
            value greater than one to parallelize the I/O. If `None`, files will be
            read sequentially.
          reader_schema: (Optional.) A `tf.string` scalar representing the reader
            schema or None.

        Raises:
          TypeError: If any argument does not have the expected type.
          ValueError: If any argument does not have the expected shape.
        """
        filenames = _create_or_validate_filenames_dataset(filenames)

        self._filenames = filenames
        self._buffer_size = buffer_size
        self._num_parallel_reads = num_parallel_reads
        self._reader_schema = reader_schema

        def creator_fn(filename):
            return _AvroRecordDataset(filename, buffer_size, reader_schema)

        self._impl = _create_dataset_reader(creator_fn, filenames, num_parallel_reads)
        variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access
        super().__init__(variant_tensor)

    def _clone(
        self,
        filenames=None,
        buffer_size=None,
        num_parallel_reads=None,
        reader_schema=None,
    ):
        return AvroRecordDataset(
            filenames or self._filenames,
            buffer_size or self._buffer_size,
            num_parallel_reads or self._num_parallel_reads,
            reader_schema or self._reader_schema,
        )

    def _inputs(self):
        return self._impl._inputs()  # pylint: disable=protected-access

    @property
    def element_spec(self):
        return tf.TensorSpec([], tf.dtypes.string)
