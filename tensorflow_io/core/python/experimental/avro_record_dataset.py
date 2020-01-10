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

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.readers import _create_or_validate_filenames_dataset, _create_dataset_reader
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow_io.core.python.ops import core_ops as avro_ops

_DEFAULT_READER_BUFFER_SIZE_BYTES = 256 * 1024  # 256 KB

# From https://github.com/tensorflow/tensorflow/blob/dc2aa7a362d579ef5fd4a1006bbccf6bebe99980/tensorflow/python/data/ops/readers.py

class _AvroRecordDataset(dataset_ops.DatasetSource):
    """A `Dataset` comprising records from one or more AvroRecord files."""

    def __init__(self, filenames, buffer_size=None, reader_schema=None):
        """Creates a `AvroRecordDataset`.
        Args:
          filenames: A `tf.string` tensor containing one or more filenames.
          buffer_size: (Optional.) A `tf.int64` scalar representing the number of
            bytes in the read buffer. 0 means no buffering.
          reader_schema: (Optional.) A `tf.string` scalar representing the reader schema or None
        """
        self._filenames = filenames
        self._buffer_size = convert.optional_param_to_tensor(
            "buffer_size",
            buffer_size,
            argument_default=_DEFAULT_READER_BUFFER_SIZE_BYTES)
        variant_tensor = avro_ops.avro_record_dataset(self._filenames,
                                                      self._buffer_size)
        super(_AvroRecordDataset, self).__init__(variant_tensor)

    @property
    def _element_structure(self):
        return structure.TensorStructure(dtypes.string, [])


#@tf_export("data.AvroRecordDataset", v1=[])
class AvroRecordDatasetV2(dataset_ops.DatasetV2):
    """A `Dataset` comprising records from one or more AvroRecord files."""

    def __init__(self,
                 filenames,
                 buffer_size=None,
                 num_parallel_reads=None,
                 reader_schema=None):
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

        self._impl = _create_dataset_reader(creator_fn, filenames,
                                            num_parallel_reads)
        variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access
        super(AvroRecordDatasetV2, self).__init__(variant_tensor)

    def _clone(self,
               filenames=None,
               buffer_size=None,
               num_parallel_reads=None):
        return AvroRecordDatasetV2(filenames or self._filenames, buffer_size or
                                 self._buffer_size, num_parallel_reads or
                                 self._num_parallel_reads)

    def _inputs(self):
        return self._impl._inputs()  # pylint: disable=protected-access

    @property
    def _element_structure(self):
        return structure.TensorStructure(dtypes.string, [])


#@tf_export(v1=["data.AvroRecordDataset"])
class AvroRecordDatasetV1(dataset_ops.DatasetV1Adapter):
    """A `Dataset` comprising records from one or more AvroRecord files."""

    def __init__(self,
                 filenames,
                 buffer_size=None,
                 num_parallel_reads=None):
        wrapped = AvroRecordDatasetV2(filenames, buffer_size,
                                    num_parallel_reads)
        super(AvroRecordDatasetV1, self).__init__(wrapped)

    __init__.__doc__ = AvroRecordDatasetV2.__init__.__doc__

    def _clone(self,
               filenames=None,
               buffer_size=None,
               num_parallel_reads=None):
        # pylint: disable=protected-access
        return AvroRecordDatasetV1(
            filenames or self._dataset._filenames,
            buffer_size or self._dataset._buffer_size,
            num_parallel_reads or self._dataset._num_parallel_reads)

    @property
    def _filenames(self):
        return self._dataset._filenames  # pylint: disable=protected-access

    @_filenames.setter
    def _filenames(self, value):
        self._dataset._filenames = value  # pylint: disable=protected-access
