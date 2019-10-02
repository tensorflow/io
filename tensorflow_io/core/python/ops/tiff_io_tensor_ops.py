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
"""TIFFIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops

class TIFFIOTensor(io_tensor_ops._CollectionIOTensor): # pylint: disable=protected-access
  """TIFFIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               internal=False):
    with tf.name_scope("TIFFIOTensor"):
      # TIFF can fit into memory so load TIFF first
      data = tf.io.read_file(filename)
      info = core_ops.decode_tiff_info(data)
      spec = tuple([
          tf.TensorSpec(tf.TensorShape(
              e.numpy().tolist() + [4]), tf.uint8) for e in info])
      columns = [i for i, _ in enumerate(spec)]
      elements = [
          io_tensor_ops.TensorIOTensor(
              core_ops.decode_tiff(data, i),
              internal=internal) for i in columns]
      super(TIFFIOTensor, self).__init__(
          spec, columns, elements,
          internal=internal)
