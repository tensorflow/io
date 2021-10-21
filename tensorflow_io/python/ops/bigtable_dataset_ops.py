from typing import List
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import tensor_spec
from tensorflow_io.python.ops import core_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf



class BigtableClient:
    """BigtableClient is the entrypoint for interacting with Cloud Bigtable in TF.

    BigtableClient encapsulates a connection to Cloud Bigtable, and exposes the
    `readSession` method to initiate a Bigtable read session.
    """

    def __init__(self, project_id, instance_id):
        """Creates a BigtableClient to start Bigtable read sessions."""
        self._client_resource = core_ops.bigtable_client(project_id, instance_id)
    
    def create_dataset(self, table_id, columns):
      return BigtableDataset(self._client_resource,  table_id, columns)



class BigtableDataset(dataset_ops.DatasetSource):
    """_BigTableDataset represents a dataset that retrieves keys and values."""

    def __init__(self, client_resource,  table_id:str, columns:List[str]):
        self._table_id = table_id
        self._columns = columns
        selected_fields = columns
        output_types = [dtypes.string]

        tensor_shapes = list(
            [] for _ in selected_fields
        )

        self._element_spec = tf.TensorSpec(shape=[len(columns)], dtype=dtypes.string)


        variant_tensor = core_ops.bigtable_dataset(client_resource, table_id, columns)
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec
