from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import tensor_spec
from tensorflow_io.python.ops import core_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

import collections


class MyDataset(dataset_ops.DatasetSource):
    """_BigQueryDataset represents a dataset that retrieves keys and values."""

    def __init__(self):
        selected_fields = ["cf1:c1"]
        output_types = [dtypes.string]

        tensor_shapes = list(
            [] for _ in selected_fields
        )

        self._element_spec = tf.TensorSpec(shape=[], dtype=dtypes.string)


        variant_tensor = core_ops.bigtable_dataset()
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec
