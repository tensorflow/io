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

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.ops import io_dataset_ops

class _AudioIODatasetFunction(object):
  def __init__(self, resource, shape, dtype):
    self._resource = resource
    self._shape = tf.TensorShape([None]).concatenate(shape[1:])
    self._dtype = dtype
  def __call__(self, start, stop):
    return core_ops.io_wav_readable_read(
        self._resource, start=start, stop=stop,
        shape=self._shape, dtype=self._dtype)

class AudioIODataset(io_dataset_ops._IODataset): # pylint: disable=protected-access
  """AudioIODataset"""

  def __init__(self,
               filename,
               internal=True):
    """AudioIODataset."""
    with tf.name_scope("AudioIODataset") as scope:
      resource = core_ops.io_wav_readable_init(
          filename,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      shape, dtype, _ = core_ops.io_wav_readable_spec(resource)
      shape = tf.TensorShape(shape.numpy())
      dtype = tf.as_dtype(dtype.numpy())
      super(AudioIODataset, self).__init__(
          _AudioIODatasetFunction(resource, shape, dtype), internal=internal)
