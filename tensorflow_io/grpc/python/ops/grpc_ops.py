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
"""GRPCInput."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.compat.v1 import data
from tensorflow_io import _load_library
grpc_ops = _load_library('_grpc_ops.so')

class GRPCDataset(data.Dataset):
  """A GRPC Dataset
  """

  def __init__(self, endpoint, shape, dtype, batch=None):
    """Create a GRPC Reader.

    Args:
      endpoint: A `tf.string` tensor containing one or more endpoints.
    """
    self._data_input = grpc_ops.grpc_input(endpoint)
    self._batch = 0 if batch is None else batch
    shape[0] = None
    self._output_shapes = tuple([
        tf.TensorShape(shape[1:])]) if self._batch == 0 else tuple([
            tf.TensorShape(shape)])
    self._output_types = tuple([dtype])
    self._batch = 0 if batch is None else batch
    super(GRPCDataset, self).__init__()

  @staticmethod
  def from_numpy(a, batch=None):
    """from_numpy"""
    from tensorflow_io.grpc.python.ops import grpc_endpoint
    grpc_server = grpc_endpoint.GRPCEndpoint(a)
    grpc_server.start()
    endpoint = grpc_server.endpoint()
    dtype = a.dtype
    shape = list(a.shape)
    dataset = GRPCDataset(endpoint, shape, dtype, batch=batch)
    dataset._grpc_server = grpc_server # pylint: disable=protected-access
    return dataset

  def __del__(self):
    if self._grpc_server is not None:
      self._grpc_server.stop()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return grpc_ops.grpc_dataset(
        self._data_input,
        self._batch,
        output_types=self.output_types,
        output_shapes=self.output_shapes)

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_classes(self):
    return tf.Tensor

  @property
  def output_types(self):
    return self._output_types
