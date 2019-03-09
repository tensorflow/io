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
"""SequenceFile Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
from tensorflow import dtypes
from tensorflow.compat.v1 import data
from tensorflow_io import _load_library
hadoop_ops = _load_library('_hadoop_ops.so')


class SequenceFileDataset(data.Dataset):
  """A Sequence File Dataset that reads the sequence file."""

  def __init__(self, filenames):
    """Create a `SequenceFileDataset`.

    `SequenceFileDataset` allows a user to read data from a hadoop sequence
    file. A sequence file consists of (key value) pairs sequentially. At
    the moment, `org.apache.hadoop.io.Text` is the only serialization type
    being supported, and there is no compression support.

    For example:

    ```python
    dataset = SequenceFileDataset("/foo/bar.seq")
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # Prints the (key, value) pairs inside a hadoop sequence file.
    while True:
      try:
        print(sess.run(next_element))
      except tf.errors.OutOfRangeError:
        break
    ```

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    self._filenames = tensorflow.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    super(SequenceFileDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return hadoop_ops.sequence_file_dataset(
        self._filenames, (dtypes.string, dtypes.string))

  @property
  def output_classes(self):
    return tensorflow.Tensor, tensorflow.Tensor

  @property
  def output_shapes(self):
    return (tensorflow.TensorShape([]), tensorflow.TensorShape([]))

  @property
  def output_types(self):
    return dtypes.string, dtypes.string
