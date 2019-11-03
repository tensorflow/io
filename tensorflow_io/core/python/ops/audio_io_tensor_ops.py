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
"""AudioIOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops

class _AudioIOTensorFunction(object):
  """_AudioIOTensorFunction"""
  def __init__(self, function, resource, shape, dtype):
    self._function = function
    self._resource = resource
    self._length = shape[0]
    self._shape = tf.TensorShape([None]).concatenate(shape[1:])
    self._dtype = dtype
  def __call__(self, start, stop):
    start, stop, _ = slice(start, stop).indices(self._length)
    if start >= self._length:
      raise IndexError("index %s is out of range" % slice(start, stop))
    return self._function(
        self._resource,
        start=start, stop=stop,
        shape=self._shape, dtype=self._dtype)
  @property
  def length(self):
    return self._length

class AudioIOTensor(io_tensor_ops.BaseIOTensor): # pylint: disable=protected-access
  """AudioIOTensor

  An `AudioIOTensor` is an `IOTensor` backed by audio files such as WAV
  format. It consists of only one `Tensor` with `shape` defined as
  `[n_samples, n_channels]`. It is a subclass of `BaseIOTensor`
  with additional `rate` property exposed, indicating the sample rate
  of the audio.
  """

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               internal=False):
    with tf.name_scope("FromAudio") as scope:
      resource = core_ops.io_wav_readable_init(
          filename,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      shape, dtype, rate = core_ops.io_wav_readable_spec(resource)
      shape = tf.TensorShape(shape.numpy())
      dtype = tf.as_dtype(dtype.numpy())
      spec = tf.TensorSpec(shape, dtype)
      function = _AudioIOTensorFunction(
          core_ops.io_wav_readable_read, resource, shape, dtype)
      self._rate = rate.numpy()
      super(AudioIOTensor, self).__init__(
          spec, function, internal=internal)

  #=============================================================================
  # Accessors
  #=============================================================================

  @io_tensor_ops._IOTensorMeta # pylint: disable=protected-access
  def rate(self):
    """The sample `rate` of the audio stream"""
    return self._rate
