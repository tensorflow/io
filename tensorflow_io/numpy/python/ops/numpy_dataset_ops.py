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
"""NumpyFile Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
numpy_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_numpy_ops.so'))


class NumpyFileDataset(dataset_ops.DatasetSource):
  """A Numpy File Dataset that reads the npy file."""

  def __init__(self, filenames, output_types=dtypes.string):
    """Create a `NumpyFileDataset`.

    `NumpyFileDataset` allows a user to read data from a numpy `npy` file.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    super(NumpyFileDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    self._output_types = output_types

  def _as_variant_tensor(self):
    return numpy_ops.numpy_file_dataset(
        self._filenames, nest.flatten(self.output_types))

  @property
  def output_classes(self):
    return nest.map_structure(lambda _: ops.Tensor, self._output_types)

  @property
  def output_shapes(self):
    return nest.map_structure(lambda _: tensor_shape.TensorShape(None),
                              self._output_types)

  @property
  def output_types(self):
    return self._output_types
