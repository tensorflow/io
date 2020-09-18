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
"""LibSVM"""

from tensorflow import sparse
from tensorflow_io.core.python.ops import core_ops


def decode_libsvm(content, num_features, dtype=None, label_dtype=None):
    """Convert Libsvm records to a tensor of label and a tensor of feature.

    Args:
      content: A `Tensor` of type `string`. Each string is a record/row in
        the Libsvm format.
      num_features: The number of features.
      dtype: The type of the output feature tensor. Default to tf.float32.
      label_dtype: The type of the output label tensor. Default to tf.int64.

    Returns:
      features: A `SparseTensor` of the shape `[input_shape, num_features]`.
      labels: A `Tensor` of the same shape as content.
    """
    labels, indices, values, shape = core_ops.io_decode_libsvm(
        content, num_features, dtype=dtype, label_dtype=label_dtype
    )
    return sparse.SparseTensor(indices, values, shape), labels


def re2_full_match(input, pattern):  # pylint: disable=redefined-builtin
    """Extract regex groups

    Args:
      input: A `tf.string` tensor
      pattern: A pattern string.
    """
    return core_ops.io_re2_full_match(input, pattern)


def read_text(filename, **kwargs):
    """read_text"""
    memory = kwargs.get("memory", "")
    offset = kwargs.get("offset", 0)
    length = kwargs.get("length", -1)
    return core_ops.io_read_text(filename, offset=offset, length=length, memory=memory)


class TextOutputSequence:
    """TextOutputSequence"""

    def __init__(self, filenames):
        """Create a `TextOutputSequence`."""
        self._filenames = filenames
        self._resource = core_ops.io_text_output_sequence(destination=filenames)

    def setitem(self, index, item):
        core_ops.io_text_output_sequence_set_item(self._resource, index, item)
