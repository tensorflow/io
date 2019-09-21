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
"""FFmpegIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import audio_io_tensor_ops

class FFmpegIOTensor(io_tensor_ops._CollectionIOTensor): # pylint: disable=protected-access
  """FFmpegIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               internal=False):
    with tf.name_scope("FFmpegIOTensor") as scope:
      from tensorflow_io.core.python.ops import ffmpeg_ops
      resource, columns = ffmpeg_ops.ffmpeg_indexable_init(
          filename,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      columns = [column.decode() for column in columns.numpy().tolist()]
      elements = []
      for column in columns:
        shape, dtype, rate = ffmpeg_ops.ffmpeg_indexable_spec(resource, column)
        shape = tf.TensorShape([None if e < 0 else e for e in shape])
        dtype = tf.as_dtype(dtype.numpy())
        spec = tf.TensorSpec(shape, dtype, column)
        if column.startswith("a:"):
          rate = rate.numpy()
          elements.append(audio_io_tensor_ops.AudioIOTensor(
              rate, spec, resource, None, partitions=None,
              internal=internal))
        else:
          elements.append(io_tensor_ops.BaseIOTensor(
              spec, resource, None, partitions=None,
              internal=internal))
      spec = tuple([e.spec for e in elements])
      super(FFmpegIOTensor, self).__init__(
          spec, columns, elements,
          internal=internal)
