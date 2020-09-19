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
"""IOTensor"""

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor
from tensorflow_io.core.python.experimental import openexr_io_tensor_ops


class IOTensor(io_tensor.IOTensor):
    """IOTensor"""

    # =============================================================================
    # Factory Methods
    # =============================================================================

    @classmethod
    def from_exr(cls, filename, **kwargs):
        """Creates an `IOTensor` from a OpenEXR file.

        Args:
          filename: A string, the filename of a OpenEXR file.
          name: A name prefix for the IOTensor (optional).

        Returns:
          A `IOTensor`.

        """
        with tf.name_scope(kwargs.get("name", "IOFromOpenEXR")):
            return openexr_io_tensor_ops.EXRIOTensor(filename, internal=True)
