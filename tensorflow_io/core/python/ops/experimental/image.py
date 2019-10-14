# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""experimental.image"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_io.core.python.ops import core_ops

def decode_tiff(contents, index=0, name=None):
  """
  Decode a TIFF-encoded image to a uint8 tensor.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The TIFF-encoded image.
    index: A `Tensor` of type int64. 0-D. The 0-based index of the frame
      inside TIFF-encoded image.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
  """
  return core_ops.decode_tiff(contents, index, name=name)
