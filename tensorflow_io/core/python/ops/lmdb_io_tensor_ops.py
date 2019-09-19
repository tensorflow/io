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
"""LMDBIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops

class LMDBIOTensor(io_tensor_ops._KeyValueIOTensor): # pylint: disable=protected-access
  """LMDBIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               internal=False):
    with tf.name_scope("LMDBIOTensor") as scope:
      resource = core_ops.lmdb_mapping_init(
          filename,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      spec = (tf.TensorSpec(tf.TensorShape([None]), tf.string),
              tf.TensorSpec(tf.TensorShape([None]), tf.string))

      class _MappingRead(object):
        def __init__(self, func):
          self._func = func
        def __call__(self, resource, key):
          return self._func(resource, key)

      class _IterableInit(object):
        def __init__(self, func, filename):
          self._func = func
          self._filename = filename
        def __call__(self):
          with tf.name_scope("IterableInit") as scope:
            return self._func(
                self._filename,
                container=scope,
                shared_name="%s/%s" % (self._filename, uuid.uuid4().hex))

      class _IterableNext(object):
        def __init__(self, func, shape, dtype):
          self._func = func
          self._shape = shape
          self._dtype = dtype
        def __call__(self, resource, capacity):
          return self._func(
              resource, capacity,
              shape=self._shape, dtype=self._dtype)

      super(LMDBIOTensor, self).__init__(
          spec,
          resource,
          _MappingRead(core_ops.lmdb_mapping_read),
          _IterableInit(
              core_ops.lmdb_iterable_init, filename),
          _IterableNext(
              core_ops.lmdb_iterable_next, tf.TensorShape([None]), tf.string),
          internal=internal)
