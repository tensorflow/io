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
"""FeatherIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops

class FeatherIOTensor(io_tensor_ops._TableIOTensor): # pylint: disable=protected-access
  """FeatherIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               internal=False):
    with tf.name_scope("FeatherIOTensor") as scope:
      resource, columns = core_ops.io_feather_readable_init(
          filename,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      columns = [column.decode() for column in columns.numpy().tolist()]
      elements = []
      for column in columns:
        shape, dtype = core_ops.io_feather_readable_spec(resource, column)
        shape = tf.TensorShape(shape.numpy())
        dtype = tf.as_dtype(dtype.numpy())
        spec = tf.TensorSpec(shape, dtype, column)
        function = io_tensor_ops._IOTensorComponentFunction( # pylint: disable=protected-access
            core_ops.io_feather_readable_read,
            resource, column, shape, dtype)
        elements.append(
            io_tensor_ops.BaseIOTensor(
                spec, function, internal=internal))
      spec = tuple([e.spec for e in elements])
      super(FeatherIOTensor, self).__init__(
          spec, columns, elements, internal=internal)
