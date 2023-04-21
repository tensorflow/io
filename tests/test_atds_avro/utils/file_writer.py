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
"""FileWriter"""

import abc
import os
import shutil
import hashlib
import tempfile
import tensorflow as tf

TF_IO_BENCHMARK_DATA_CACHE = "TF_IO_BENCHMARK_DATA_CACHE"


class FileWriter(metaclass=abc.ABCMeta):
    """Base class for file writer.

    FileWriter consumes a DataSource and generates benchmark data as described
    in the DataSource. DataSource contains benchmark data metadata such as value
    generators, total number of records, and number of partitioned files.

    SubClass must implement `_write_to_path` function to write data under the
    given path and follow the spec described in DataSource.
    """

    def __init__(self):
        """Create a new FileWriter.

        This must be called by the constructors of subclasses.
        """
        self._data_cache_path = os.getenv(TF_IO_BENCHMARK_DATA_CACHE, None)
        self._dir_path = self._data_cache_path

    def __enter__(self):
        """Enter a context to create dir_path for file generation."""
        if not self._data_cache_path:
            self._dir_path = tempfile.mkdtemp()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up dir_path after exiting the context."""
        if not self._data_cache_path:
            shutil.rmtree(self._dir_path, ignore_errors=True)
        self._dir_path = None

    def write(self, data_source):
        """Generate benchmark data and return the path to the generated files.

        Args:
          data_source: A DataSource object that describes the properties of
            the benchmark data.

        Returns:
          A str path to the generated files.
        """
        # Use data source hash code as data source folder name.
        data_source_path = data_source.hash_code()
        # Use writer hash code as the writer folder path.
        writer_dir = self.hash_code()

        if self._data_cache_path is not None:
            cached_dataset = None
            writer_path = os.path.join(
                self._data_cache_path, data_source_path, writer_dir
            )
            if os.path.exists(writer_path):
                return writer_path
            from tests.test_atds_avro.utils.tf_record_writer import (
                TFRecordWriter,
            )

            with TFRecordWriter() as tf_writer:
                tf_record_cache_dir = os.path.join(
                    self._data_cache_path, data_source_path, tf_writer.hash_code()
                )
                if not os.path.exists(tf_record_cache_dir):
                    os.makedirs(tf_record_cache_dir, exist_ok=True)
                    tf_writer._write_to_path(tf_record_cache_dir, data_source)
                parser_fn = tf_writer.create_tf_example_parser_fn(data_source)
                pattern = os.path.join(tf_record_cache_dir, f"*.{tf_writer.extension}")
                cached_dataset = tf.data.Dataset.list_files(pattern, shuffle=False)
                cached_dataset = tf.data.TFRecordDataset(cached_dataset)
                cached_dataset = cached_dataset.map(parser_fn)
                self._write_to_path_from_cached_data(
                    writer_path, data_source, cached_dataset
                )
                return writer_path
        else:
            writer_path = os.path.join(self._dir_path, data_source_path, writer_dir)
            if not os.path.exists(writer_path):
                os.makedirs(writer_path, exist_ok=True)
                self._write_to_path(writer_path, data_source)
            return writer_path

    @abc.abstractmethod
    def _write_to_path(self, dir_path, data_source):
        """Generate benchmark data and write the data under the given path.

        Args:
          dir_path: A str path to write files to.
          data_source: A DataSource object.

        Raises:
          NotImplementedError: If subclass does not overload the function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _write_to_path_from_cached_data(self, dir_path, data_source, dataset):
        """Write the given dataset to the given path.

        Args:
          dir_path: A str path to write dataset to.
          data_source: A DataSource object.
          dataset: Cached dataset containing data to write.

        Raises:
          NotImplementedError: If subclass does not overload the function.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def extension(self):
        """Returns the extension of the filename e.g. tfrecords, avro, etc.

        Raises:
          NotImplementedError: If subclass does not overload the function.
        """
        raise NotImplementedError

    def hash_code(self):
        """Return the hashed code of this file writer.

        The hashed code is used to create the folder that the writer can write
        data to. It is useful for benchmark data cache.

        Returns:
          The hashed code of the writer in hex str.
        """
        m = hashlib.sha256()
        # Hash the instance class name by default. Subclass can overload the
        # function to support customized hashing logic for its own state.
        m.update(self.__class__.__name__.encode())
        return m.hexdigest()

    def _get_filenames_to_num_records(self, data_source):
        """Returns a dict mapping filenames to the number of records in that file.

        Args:
          data_source: A DataSource describing the data to be written.

        Returns:
          A dict mapping filename to number of records in that file.
        """
        filenames_to_num_records = {}
        partitions = data_source.partitions
        record_per_partition = data_source.num_records // partitions
        remaining = data_source.num_records % partitions

        partition_length = len(str(partitions))
        for file_index in range(partitions):
            # Add leading zero to index_name e.g. 0001
            index_name = str(file_index).zfill(partition_length)
            filename = f"part-{index_name}.{self.extension}"

            num_records = record_per_partition
            if remaining and file_index < remaining:
                num_records = num_records + 1
            filenames_to_num_records[filename] = num_records
        return filenames_to_num_records
