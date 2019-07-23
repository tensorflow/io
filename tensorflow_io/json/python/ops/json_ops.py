# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""JSONDataset"""
import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops
from tensorflow_io.core.python.ops import core_ops as json_ops

class JSONDataset(data_ops.Dataset):
  """A JSONLabelDataset. JSON (JavaScript Object Notation) is a lightweight data-interchange format.
  """

  def __init__(self, filenames, columns, dtypes, batch=None):
    """Create a JSONLabelDataset.

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      columns: A 0-D or 1-D `tf.int32` tensor containing the columns to extract.
      dtypes: A tuple of `tf.DType` objects representing the types of the
        columns returned.
    """
    data_input = json_ops.json_input(
        filenames, ["none", "gz"], columns=columns)
    dtypes = dtypes
    batch = 0 if batch is None else batch
    shapes = [
        tf.TensorShape([]) for _ in columns] if batch == 0 else [
            tf.TensorShape([None]) for _ in columns]
    super(JSONDataset, self).__init__(
        json_ops.json_dataset,
        data_input,
        batch, dtypes, shapes
    )
