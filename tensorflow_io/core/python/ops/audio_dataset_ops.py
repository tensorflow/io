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
"""AudioDataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

class AudioGraphIODataset(tf.data.Dataset):
  """AudioGraphIODataset"""

  def __init__(self,
               resource,
               shape, dtype,
               internal=True):
    """AudioGraphIODataset."""
    with tf.name_scope("AudioGraphIODataset"):
      assert internal

      capacity = 1024 #kwargs.get("capacity", 4096)

      self._resource = resource
      dataset = tf.data.Dataset.range(0, shape[0], capacity)
      dataset = dataset.map(lambda index: core_ops.io_audio_readable_read(
          resource, index, index+capacity, dtype=dtype))
      dataset = dataset.apply(
          tf.data.experimental.take_while(
              lambda v: tf.greater(tf.shape(v)[0], 0)))
      dataset = dataset.unbatch()
      self._dataset = dataset
      super(AudioGraphIODataset, self).__init__(
          self._dataset._variant_tensor) # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec

class AudioIODataset(AudioGraphIODataset):
  """AudioIODataset"""

  def __init__(self,
               filename,
               internal=True):
    """AudioIODataset."""
    with tf.name_scope("FromAudio"):
      resource = core_ops.io_audio_readable_init(filename)
      shape, dtype, _ = core_ops.io_audio_readable_spec(resource)
      shape = tf.TensorShape(shape)
      dtype = tf.as_dtype(dtype.numpy())
      super(AudioIODataset, self).__init__(
          resource, shape, dtype, internal=internal)
