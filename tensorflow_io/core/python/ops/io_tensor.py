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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

class IOTensor(object):
  """IOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               resource,
               dtype,
               shape,
               internal=False):
    """Creates an `IOTensor`.

    This constructor is private -- please use one of the following ops to
    build `IOTensor`s:

      * `tfio.IOTensor.from_tensor`
      * `tfio.IOTensor.from_audio`

    Args:
      resource: A resource in tensorflow.
      dtype: The type of the tensor.
      shape: The shape of the tensor.
      internal: True if the constructor is being called by one of the factory
        methods.  If false, an exception will be raised.

    Raises:
      TypeError:
    """
    if not internal:
      raise ValueError("IOTensor constructor is private; please use one "
                       "of the factory methods instead (e.g., "
                       "IOTensor.from_tensor())")
    self._resource = resource
    self._dtype = tf.as_dtype(dtype.numpy())
    self._shape = tf.TensorShape([
        None if dim < 0 else dim for dim in shape.numpy() if dim != 0])

  #=============================================================================
  # Factory Methods
  #=============================================================================

  @classmethod
  def from_audio(cls,
                 filename,
                 **kwargs):
    """Creates an `IOTensor` from an audio file.

    The following audio file formats are supported:
    - WAV

    Args:
      filename: A string, the filename of an audio file.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IOTensor`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromAudio")) as scope:
      resource, dtypes, shapes, rate = core_ops.wav_indexable_init(
          filename, memory="", metadata="",
          container=scope, shared_name=filename)
      return AudioIOTensor(
          resource, dtypes[0], shapes[0], rate, internal=True)

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def dtype(self):
    """The `DType` of values in this tensor."""
    return self._dtype

  @property
  def shape(self):
    """The statically known shape of this io tensor.

    Returns:
      A `TensorShape` containing the statically known shape of this io
      tensor. The first dimension could have a size of `None` if this
      io tensor is from an iterable.

    Examples:

      ```python
      ```
    """
    return self._shape

  @property
  def rank(self):
    """The number of dimensions in this io tensor.

    Returns:
      A Python `int` indicating the number of dimensions in this io
      tensor.
    """
    return tf.rank(self._shape)


  #=============================================================================
  # Tensor Type Conversions
  #=============================================================================

  @classmethod
  def from_tensor(cls,
                  tensor,
                  **kwargs):
    """Converts a `tf.Tensor` into a `IOTensor`.

    Examples:

    ```python
    ```

    Args:
      tensor: The `Tensor` to convert.

    Returns:
      A `IOTensor`.

    Raises:
      ValueError: If tensor is not a `Tensor`.
    """
    with tf.name_scope(kwargs.get("name", "IOFromTensor")):
      _ = tensor
      raise NotImplementedError()

  def to_tensor(self, **kwargs):
    """Converts this `IOTensor` into a `tf.Tensor`.

    Example:

    ```python
    ```

    Args:
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `Tensor` with value obtained from this `IOTensor`.
    """
    with tf.name_scope(kwargs.get("name", "IOToTensor")):
      return self.__getitem__(slice(None, None))

  #=============================================================================
  # String Encoding
  #=============================================================================
  def __repr__(self):
    return "<tfio.IOTensor: shape=%s, dtype=%s>" % (self.shape, self.dtype.name)

  #=============================================================================
  # Indexing & Slicing
  #=============================================================================
  def __getitem__(self, key):
    """Returns the specified piece of this IOTensor."""
    if isinstance(key, slice):
      start = key.start
      stop = key.stop
      step = key.step
      if start is None:
        start = 0
      if stop is None:
        stop = -1
      if step is None:
        step = 1
    else:
      start = key
      stop = key + 1
      step = 1
    return self._getitem_func(
        self._resource, start, stop, step, dtype=self.dtype)

  def __len__(self):
    """Returns the total number of items of this IOTensor."""
    return abs(self.shape[0])


class AudioIOTensor(IOTensor):
  """AudioIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               resource,
               dtype,
               shape,
               rate,
               internal=False):
    self._rate = rate.numpy()
    self._getitem_func = core_ops.wav_indexable_get_item
    super(AudioIOTensor, self).__init__(
        resource, dtype, shape, internal=internal)

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def rate(self):
    """The sampel `rate` of the audio stream"""
    return self._rate

  #=============================================================================
  # String Encoding
  #=============================================================================
  def __repr__(self):
    return "<tfio.AudioIOTensor: shape=%s, dtype=%s, rate=%s>" % (
        self.shape, self.dtype.name, self.rate)
