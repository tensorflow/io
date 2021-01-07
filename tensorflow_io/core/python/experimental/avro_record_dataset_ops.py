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


def _require(condition: bool, err_msg: str = None) -> None:
    """Checks if the specified condition is true else raises exception
    
    Args:
        condition: The condition to test
        err_msg: If specified, it's the error message to use if condition is not true.

    Raises:
      ValueError: Raised when the condition is false

    Returns:
        None
    """
    if not condition:
        raise ValueError(err_msg)


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
def _create_dataset_reader(
    dataset_creator,
    filenames,
    cycle_length=None,
    num_parallel_calls=None,
    deterministic=None,
    block_length=1,
):
    """
    This creates a dataset reader which reads records from multiple files and interleaves them together
```
dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
# NOTE: New lines indicate "block" boundaries.
dataset = dataset.interleave(
    lambda x: Dataset.from_tensors(x).repeat(6),
    cycle_length=2, block_length=4)
list(dataset.as_numpy_iterator())
```
Results in the following output:
[1,1,1,1,
 2,2,2,2,
 1,1,
 2,2,
 3,3,3,3,
 4,4,4,4,
 3,4,
 5,5,5,5,
 5,5,
]
    Args:
        dataset_creator: Initializer for AvroDatasetRecord
        filenames: A `tf.data.Dataset` iterator of filenames to read
        cycle_length: The number of files to be processed in parallel. This is used by `Dataset.Interleave`.
        We set this equal to `block_length`, so that each time n number of records are returned for each of the n
        files.
        num_parallel_calls: Number of threads spawned by the interleave call.
        deterministic: Sets whether the interleaved records are written in deterministic order. in tf.interleave this is default true
        block_length: Sets the number of output on the output tensor. Defaults to 1
    Returns:
        A dataset iterator with an interleaved list of parsed avro records.

    """

    def read_many_files(filenames):
        filenames = tf.convert_to_tensor(filenames, tf.string, name="filename")
        return dataset_creator(filenames)

    if cycle_length is None:
        return filenames.flat_map(read_many_files)

    return filenames.interleave(
        read_many_files,
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls,
        block_length=block_length,
        deterministic=deterministic,
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
        self,
        filenames,
        buffer_size=None,
        num_parallel_reads=None,
        num_parallel_calls=None,
        reader_schema=None,
        deterministic=True,
        block_length=1,
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
            read sequentially. This must be set to equal or greater than `num_parallel_calls`.
            This constraint exists because `num_parallel_reads` becomes `cycle_length` in the
            underlying call to `tf.Dataset.Interleave`, and the `cycle_length` is required to be
            equal or higher than the number of threads(`num_parallel_calls`).
            `cycle_length` in tf.Dataset.Interleave will dictate how many items it will pick up to process
          num_parallel_calls: (Optional.)  number of thread to spawn. This must be set to `None`
            or greater than 0. Also this must be less than or equal to `num_parallel_reads`. This defines
            the degree of parallelism in the underlying Dataset.interleave call.
          reader_schema: (Optional.) A `tf.string` scalar representing the reader
            schema or None.
          deterministic: (Optional.) A boolean controlling whether determinism should be traded for performance by
          allowing elements to be produced out of order. Defaults to `True`
          block_length: Sets the number of output on the output tensor. Defaults to 1
        Raises:
          TypeError: If any argument does not have the expected type.
          ValueError: If any argument does not have the expected shape.
        """
        _require(
            num_parallel_calls is None
            or num_parallel_calls == tf.data.experimental.AUTOTUNE
            or num_parallel_calls > 0,
            f"num_parallel_calls: {num_parallel_calls} must be set to None, "
            f"tf.data.experimental.AUTOTUNE, or greater than 0",
        )
        if num_parallel_calls is not None:
            _require(
                num_parallel_reads is not None
                and (
                    num_parallel_reads >= num_parallel_calls
                    or num_parallel_reads == tf.data.experimental.AUTOTUNE
                ),
                f"num_parallel_reads: {num_parallel_reads} must be greater than or equal to "
                f"num_parallel_calls: {num_parallel_calls} or set to tf.data.experimental.AUTOTUNE",
            )

        filenames = _create_or_validate_filenames_dataset(filenames)

        self._filenames = filenames
        self._buffer_size = buffer_size
        self._num_parallel_reads = num_parallel_reads
        self._num_parallel_calls = num_parallel_calls
        self._reader_schema = reader_schema
        self._block_length = block_length

        def read_multiple_files(filenames):
            return _AvroRecordDataset(filenames, buffer_size, reader_schema)

        self._impl = _create_dataset_reader(
            read_multiple_files,
            filenames,
            cycle_length=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            deterministic=deterministic,
            block_length=block_length,
        )
        variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access
        super().__init__(variant_tensor)

    def _clone(
        self,
        filenames=None,
        buffer_size=None,
        num_parallel_reads=None,
        num_parallel_calls=None,
        reader_schema=None,
        block_length=None,
    ):
        return AvroRecordDataset(
            filenames or self._filenames,
            buffer_size or self._buffer_size,
            num_parallel_reads or self._num_parallel_reads,
            num_parallel_calls or self._num_parallel_calls,
            reader_schema or self._reader_schema,
            block_length or self._block_length,
        )

    def _inputs(self):
        return self._impl._inputs()  # pylint: disable=protected-access

    @property
    def element_spec(self):
        return tf.TensorSpec([], tf.dtypes.string)
