# Copyright 2021 Google LLC
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

from typing import List
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import tensor_spec
from tensorflow_io.python.ops import core_ops
import tensorflow_io.python.ops.bigtable.bigtable_version_filters as filters
import tensorflow_io.python.ops.bigtable.bigtable_row_set as bigtable_row_set
import tensorflow_io.python.ops.bigtable.bigtable_row_range as bigtable_row_range
from tensorflow.python.framework import dtypes
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops


class BigtableClient:
    """BigtableClient is the entrypoint for interacting with Cloud Bigtable in TF.

    BigtableClient encapsulates a connection to Cloud Bigtable, and exposes the
    `readSession` method to initiate a Bigtable read session.
    """

    def __init__(self, project_id: str, instance_id: str):
        """Creates a BigtableClient to start Bigtable read sessions."""
        self._client_resource = core_ops.bigtable_client(project_id, instance_id)

    def get_table(self, table_id):
        return BigtableTable(self._client_resource, table_id)


class BigtableTable:
    """Entry point for reading data from Cloud Bigtable. This object represents
    a Bigtable Table and provides basic methods for reading from it.
    """

    def __init__(self, client_resource: tf.Tensor, table_id: str):
        """
        Args:
            client_resource: Resource holding a reference to BigtableClient.
            table_id (str): The ID of the table.
        """
        self._table_id = table_id
        self._client_resource = client_resource

    def read_rows(
        self,
        columns: List[str],
        row_set: bigtable_row_set.RowSet,
        filter: filters.BigtableFilter = None,
        output_type=tf.string,
    ):
        """Retrieves values from Google Bigtable sorted by RowKeys.
        Args:
            columns (List[str]): the list of columns to read from; the order on
                this list will determine the order in the output tensors
            row_set (RowSet): set of rows to read.

        Returns:
            A `tf.data.Dataset` returning the cell contents.
        """

        # Python initializes the default arguments once at the start of the
        # program. If the fork happens after that (for instance when we run
        # tests using xdist) the program deadlocks and hangs. That is why we
        # have to make sure, all default arguments are initialized on each
        # invocation.
        if filter is None:
            filter = filters.latest()
        return _BigtableDataset(
            self._client_resource, self._table_id, columns, row_set, filter, output_type
        )

    def parallel_read_rows(
        self,
        columns: List[str],
        num_parallel_calls=tf.data.AUTOTUNE,
        row_set: bigtable_row_set.RowSet = None,
        filter: filters.BigtableFilter = None,
        output_type=tf.string,
    ):
        """Retrieves values from Google Bigtable in parallel. The ammount of work
        is split between workers based on SampleRowKeys. Keep in mind that when
        reading in parallel, rows are not read in any particular order.
        Args:
            columns (List[str]): the list of columns to read from; the order on
                this list will determine the order in the output tensors
            num_parallel_calls: number of workers assigned to reading the data.
            row_set (RowSet): set of rows to read.

        Returns:
            A `tf.data.Dataset` returning the cell contents.
        """

        # We have to make sure that all the default arguments are initialized
        # on each invocation. For more info see read_rows method.
        if row_set is None:
            row_set = bigtable_row_set.from_rows_or_ranges(
                bigtable_row_range.infinite()
            )
        if filter is None:
            filter = filters.latest()

        samples = core_ops.bigtable_split_row_set_evenly(
            self._client_resource, row_set._impl, self._table_id, num_parallel_calls
        )

        def map_func(idx):
            return self.read_rows(
                columns, bigtable_row_set.RowSet(samples[idx]), filter, output_type
            )

        # We interleave a dataset of sample's indexes instead of a dataset of
        # samples, because Dataset.from_tensor_slices attempts to copy the
        # resource tensors using DeepCopy from tensor_util.cc which is not
        # possible for tensors of type DT_RESOURCE.
        return tf.data.Dataset.range(samples.shape[0]).interleave(
            map_func=map_func,
            cycle_length=num_parallel_calls,
            block_length=1,
            num_parallel_calls=num_parallel_calls,
            deterministic=False,
        )


class _BigtableDataset(dataset_ops.DatasetSource):
    """_BigtableDataset represents a dataset that retrieves keys and values."""

    def __init__(
        self,
        client_resource,
        table_id: str,
        columns: List[str],
        row_set: bigtable_row_set.RowSet,
        filter,
        output_type,
    ):
        self._table_id = table_id
        self._columns = columns
        self._filter = filter
        self._element_spec = tf.TensorSpec(shape=[len(columns)], dtype=output_type)

        variant_tensor = core_ops.bigtable_dataset(
            client_resource, row_set._impl, filter._impl, table_id, columns, output_type
        )
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec
