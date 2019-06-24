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
"""TextInput/TextOutput."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops as data_ops
from tensorflow_io.core.python.ops import core_ops as text_ops

class TextDataset(data_ops.Dataset):
  """A Text Dataset"""

  def __init__(self, filename, batch=None):
    """Create a Text Reader.

    Args:
      filename: A `tf.string` tensor containing one or more filenames.
    """
    batch = 0 if batch is None else batch
    dtypes = [tf.string]
    shapes = [
        tf.TensorShape([])] if batch == 0 else [
            tf.TensorShape([None])]
    super(TextDataset, self).__init__(
        text_ops.text_dataset,
        text_ops.text_input(filename, ["none", "gz"]),
        batch, dtypes, shapes)


class TextOutputSequence(object):
  """TextOutputSequence"""

  def __init__(self, filenames):
    """Create a `TextOutputSequence`.
    """
    self._filenames = filenames
    self._resource = text_ops.text_output_sequence(destination=filenames)

  def setitem(self, index, item):
    text_ops.text_output_sequence_set_item(self._resource, index, item)
