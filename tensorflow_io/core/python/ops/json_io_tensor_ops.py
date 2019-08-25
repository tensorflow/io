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
"""JSONIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops

class JSONIOTensor(io_tensor_ops._TableIOTensor): # pylint: disable=protected-access
  """JSONIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               mode=None,
               internal=False):
    with tf.name_scope("JSONIOTensor") as scope:
      metadata = [] if mode is None else ["mode: %s" % mode]
      resource, shapes, dtypes, columns = core_ops.json_indexable_init(
          filename,
          metadata=metadata,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      shapes = [
          tf.TensorShape(
              [None if dim < 0 else dim for dim in e.numpy() if dim != 0]
          ) for e in tf.unstack(shapes)]
      dtypes = [tf.as_dtype(e.numpy()) for e in tf.unstack(dtypes)]
      columns = [e.numpy().decode() for e in tf.unstack(columns)]
      spec = tuple([tf.TensorSpec(shape, dtype, column) for (
          shape, dtype, column) in zip(shapes, dtypes, columns)])
      super(JSONIOTensor, self).__init__(
          spec, columns,
          resource, core_ops.json_indexable_get_item,
          internal=internal)
