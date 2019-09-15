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
"""_IOStorage"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

class _IOStorageMeta(property):
  """_IOStorageMeta is a decorator that is viewable to __repr__"""
  pass

class _IOStorage(object):
  """_IOStorage"""

  def __init__(self,
               storage,
               internal=False):
    if not internal:
      raise ValueError("IOStorage constructor is private; please use one "
                       "of the factory methods instead (e.g., "
                       "IOStorage.from_s3())")
    self._storage = storage
    super(_IOStorage, self).__init__()

  #=============================================================================
  # Accessors
  #=============================================================================

  def list(self):
    """The list of the entries in the storage."""
    return self._storage

  #=============================================================================
  # String Encoding
  #=============================================================================
  def __repr__(self):
    meta = "".join([", %s=%s" % (
        k, repr(v.__get__(self))) for k, v in self.__class__.__dict__.items(
            ) if isinstance(v, _IOStorageMeta)])
    return "<%s: %s>" % (
        self.__class__.__name__, meta)

  #=============================================================================
  # Iterator
  #=============================================================================
  def __iter__(self):
    with tf.name_scope("IOStorageIter"):
      for e in tf.unstack(self._storage):
        yield e

  #=============================================================================
  # Batch operations
  #=============================================================================

  def read(self, offset=None, length=None):
    """Read data into a 1-D string Tensor.

    Example:

    ```python
    ```

    Args:
      offset: A 0-D or 1-D int64 Tensor specifying the offset of the data.
      length: A 0-D or 1-D int64 Tensor specifying the length of the data.

    Returns:
      A 1-D string Tensor.
    """
    offset = [] if offset is None else offset
    length = [] if length is None else length
    return core_ops.storage_read(self._storage, offset, length)

  def size(self):
    """Get the size of the stoarge into a 1-D int64 Tensor.

    Example:

    ```python
    ```

    Args:

    Returns:
      A 1-D int64 Tensor.
    """
    return core_ops.storage_size(self._storage)

class S3IOStorage(_IOStorage):
  """S3IOStorage
  """

  def __init__(self,
               prefix,
               internal=False):
    with tf.name_scope("S3IOStorage"):
      storage = core_ops.storage_list_s3(prefix)
      super(S3IOStorage, self).__init__(
          storage,
          internal=internal)
