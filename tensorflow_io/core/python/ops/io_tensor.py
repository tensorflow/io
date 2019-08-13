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
from tensorflow.python.framework import composite_tensor
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.ops import archive_ops

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
      * `tfio.IOTensor.from_mnist`

    Args:
      values: A potentially ragged tensor of any dtype and shape `[nvals, ...]`.
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
    self._dtype = dtype
    self._shape = shape

  #=============================================================================
  # Factory Methods
  #=============================================================================

  @classmethod
  def from_mnist(cls,
                 filename,
                 **kwargs):
    """Creates an `IOTensor` from either MNIST labels or images file `filename`.

    Args:
      filename: A string, the filename of either a MNIST labels or images file.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IOTensor`.

    Raises:
      TypeError: If `value` is not a `Tensor`.

    """
    with tf.name_scope(kwargs.get("name", None), "IOFromMNIST",
                       [filename]) as scope:
      f, entries = archive_ops.list_archive_entries(
          filename, ["none", "gz"])
      # TODO: In eager mode, we should be able to find out
      # if the file is compressed or not, and skip archive
      # process if not needed
      memory = archive_ops.read_archive(filename, f, entries)
      resource, dtype, shape = core_ops.init_mnist(
        filename, memory=memory, metadata="", container=scope, shared_name=filename)
      v = cls(resource, dtype, shape, internal=True)
      # Reference memory as otherwise it will be freed.
      v._memory = memory
      return v

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def dtype(self):
    """The `DType` of values in this tensor."""
    return self._dtype.numpy()

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
    return tensor_shape.TensorShape([None]).concatenate(self._shape[1:].numpy())

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
    with tf.name_scope(name, "IOFromTensor", [tensor]):
      tensor = ops.convert_to_tensor(tensor, name="tensor")
      return None

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
    with tf.name_scope(kwargs.get("name", None), "IOToTensor",
                       [self]):
      return self[:]

  #=============================================================================
  # String Encoding
  #=============================================================================
  def __repr__(self):
    return "<tf.IOTensor %s>" % self.to_list()

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
    return core_ops.get_item_mnist(self._resource, start, stop, step)

  def __len__(self):
    """Returns the total number of items of this IOTensor."""
    return core_ops.len_mnist(self._resource)

  #=============================================================================
  # Composite Tensor
  #=============================================================================
