from typing import List
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import tensor_spec
from tensorflow_io.python.ops import core_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops

from tensorflow_io.python.ops.bigtable.bigtable_row_set import (
    from_rows_or_ranges,
    RowSet,
    intersect,
)
from tensorflow_io.python.ops.bigtable.bigtable_row_range import infinite


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
    def __init__(self, client_resource, table_id: str):
        self._table_id = table_id
        self._client_resource = client_resource

    def read_rows(self, columns: List[str], row_set: RowSet):
        return _BigtableDataset(self._client_resource, self._table_id, columns, row_set)

    def parallel_read_rows(
        self,
        columns: List[str],
        num_parallel_calls=tf.data.AUTOTUNE,
        row_set: RowSet = from_rows_or_ranges(infinite()),
    ):

        print("calling parallel read_rows with row_set:", row_set)
        samples = core_ops.bigtable_split_row_set_evenly(
            self._client_resource, row_set._impl, self._table_id, num_parallel_calls,
        )

        def map_func(idx):
            return self.read_rows(columns, RowSet(samples[idx]))

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
        self, client_resource, table_id: str, columns: List[str], row_set: RowSet,
    ):
        self._table_id = table_id
        self._columns = columns
        self._element_spec = tf.TensorSpec(shape=[len(columns)], dtype=dtypes.string)

        variant_tensor = core_ops.bigtable_dataset(
            client_resource, row_set._impl, table_id, columns
        )
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec
