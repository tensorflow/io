from typing import List
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import tensor_spec
from tensorflow_io.python.ops import core_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

import collections


class MyDataset(dataset_ops.DatasetSource):
    """_BigQueryDataset represents a dataset that retrieves keys and values."""

    def __init__(self, project_id:str, instance_id:str, table_id:str, columns:List[str]):
        self._project_id = project_id
        self._instance_id = instance_id
        self._table_id = table_id
        self._columns = columns
        selected_fields = columns
        output_types = [dtypes.string]

        tensor_shapes = list(
            [] for _ in selected_fields
        )

        self._element_spec = tf.TensorSpec(shape=[len(columns)], dtype=dtypes.string)


        variant_tensor = core_ops.bigtable_dataset(project_id, instance_id, table_id, columns)
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec
