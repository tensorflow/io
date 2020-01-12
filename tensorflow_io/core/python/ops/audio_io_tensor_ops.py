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

import tensorflow as tf
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import core_ops

class AudioGraphIOTensor():
  """AudioGraphIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               resource,
               shape, dtype, rate,
               internal=False):
    with tf.name_scope("AudioGraphIOTensor"):
      assert internal
      self._resource = resource
      self._shape = shape
      self._dtype = dtype
      self._rate = rate
      super(AudioGraphIOTensor, self).__init__()

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def shape(self):
    """Returns the `TensorShape` that represents the shape of the tensor."""
    return self._shape

  @property
  def dtype(self):
    """Returns the `dtype` of elements in the tensor."""
    return self._dtype

  #=============================================================================
  # String Encoding
  #=============================================================================
  def __repr__(self):
    meta = "".join([", %s=%s" % (
        k, repr(v.__get__(self))) for k, v in self.__class__.__dict__.items(
            ) if isinstance(v, io_tensor_ops._IOTensorMeta)]) # pylint: disable=protected-access
    return "<%s: shape=%s, dtype=%s | %s>" % (
        self.__class__.__name__, self.shape, self.dtype, meta)

  #=============================================================================
  # Tensor Type Conversions
  #=============================================================================

  def to_tensor(self):
    """Converts this `IOTensor` into a `tf.Tensor`.

    Args:
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `Tensor` with value obtained from this `IOTensor`.
    """
    return core_ops.io_audio_readable_read(
        self._resource, 0, -1, dtype=self._dtype)

  #=============================================================================
  # Indexing and slicing
  #=============================================================================
  def __getitem__(self, key):
    """Returns the specified piece of this IOTensor."""
    if isinstance(key, slice):
      return core_ops.io_audio_readable_read(
          self._resource, key.start, key.stop, dtype=self._dtype)
    item = core_ops.io_audio_readable_read(
        self._resource, key, key + 1, dtype=self._dtype)
    if tf.shape(item)[0] == 0:
      raise IndexError("index %s is out of range" % key)
    return item[0]

  def __len__(self):
    """Returns the total number of items of this IOTensor."""
    return self._shape[0]

  #=============================================================================
  # Accessors
  #=============================================================================
  @io_tensor_ops._IOTensorMeta # pylint: disable=protected-access
  def rate(self):
    """The sample `rate` of the audio stream"""
    return self._rate

class AudioIOTensor(AudioGraphIOTensor):
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
    with tf.name_scope("FromAudio"):
      resource = core_ops.io_audio_readable_init(filename)
      shape, dtype, rate = core_ops.io_audio_readable_spec(resource)
      shape = tf.TensorShape(shape)
      dtype = tf.as_dtype(dtype.numpy())
      super(AudioIOTensor, self).__init__(
          resource, shape, dtype, rate, internal=internal)
