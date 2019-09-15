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
"""IOStorage"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import io_storage_ops

class IOStorage(io_storage_ops._IOStorage):  # pylint: disable=protected-access
  """IOStorage

  An `IOStorage` is a collection of objects in storages such as AWS S3 or
  Google Cloud GCS. It is capable of reading storage in batch operations
  so that objects could be populated into a 1-D string tensor.
  This could be helpful in situations such as checking the magic
  number (normally the first several bytes) to identify the file types.

  """

  #=============================================================================
  # Factory Methods
  #=============================================================================

  @classmethod
  def from_s3(cls,
              prefix,
              **kwargs):
    """Construct a S3IOStorage from a prefix.

    Examples:

    ```python
    ```

    Args:
      prefix: The prefix of the s3 objects.

    Returns:
      A `S3IOStorage`.

    Raises:
      ValueError: If tensor is not a `Tensor`.
    """
    with tf.name_scope(kwargs.get("name", "IOStorageFromS3")):
      return io_storage_ops.S3IOStorage(prefix, internal=True)
