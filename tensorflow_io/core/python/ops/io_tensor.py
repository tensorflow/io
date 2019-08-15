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

import collections

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.core.python.ops import data_ops
from tensorflow_io.kafka.python.ops.kafka_ops import kafka_ops

class _IOBaseTensor(object):
  """_IOBaseTensor"""

  def __init__(self,
               dtype,
               shape,
               properties,
               internal=False):
    if not internal:
      raise ValueError("IOTensor constructor is private; please use one "
                       "of the factory methods instead (e.g., "
                       "IOTensor.from_tensor())")
    self._dtype = dtype
    self._shape = shape
    self._properties = collections.OrderedDict(
        {} if properties is None else properties)
    super(_IOBaseTensor, self).__init__()

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

  @property
  def properties(self):
    """The properties associated with this tensor.

    Returns:
      A ordered dict with name and properties associated with this tensor.
    """
    return self._properties

  #=============================================================================
  # String Encoding
  #=============================================================================
  def __repr__(self):
    props = "".join([
        ", %s: %s" % (k, repr(v)) for (k, v) in self.properties.items()])
    return "<%s: shape=%s, dtype=%s%s>" % (
        type(self).__name__, self.shape, self.dtype.name, props)


class IOTensor(_IOBaseTensor):
  """IOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               dtype,
               shape,
               resource,
               function,
               properties,
               internal=False):
    """Creates an `IOTensor`.

    This constructor is private -- please use one of the following ops to
    build `IOTensor`s:

      * `tfio.IOTensor.from_tensor`
      * `tfio.IOTensor.from_audio`

    Args:
      dtype: The type of the tensor.
      shape: The shape of the tensor.
      resource: The resource associated with the IO.
      function: The function for indexing and accessing items with resource.
      properties: An ordered dict of properties to be printed.
      internal: True if the constructor is being called by one of the factory
        methods.  If false, an exception will be raised.

    Raises:
      TypeError:
    """
    self._resource = resource
    self._function = function
    super(IOTensor, self).__init__(dtype, shape, properties, internal=internal)

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
    return self._function(
        self._resource, start, stop, step, dtype=self.dtype)

  def __len__(self):
    """Returns the total number of items of this IOTensor."""
    return abs(self.shape[0])

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
    with tf.name_scope(kwargs.get("name", "IOFromAudio")):
      return AudioIOTensor(filename, internal=True)

  @classmethod
  def from_kafka(cls,
                 subscription,
                 servers=None,
                 timeout=None,
                 eof=True,
                 conf=None,
                 **kwargs):
    """Creates an `IOTensor` from a Kafka stream.

    Args:
      subscription: A string specifying the topic and partition for
        Kafka stream server. The subscription is in the format
        of [topic:partition:start:end], where start is
        the start offset and end is the end offset.
        By default start will be 0 and end will be -1.
      servers: A string specifying bootstrap severs of the Kafka.
        The default is `localhost:9092`.
      timeout: An int64 for consumer timeout (in milliseconds).
      eof: If True, the kafka reader will stop on EOF.
      conf: A list of strings to specify the configurations.
        All topic configurations will be prefixed with `topic.`.
        e.g., `topic.compression.type=gzip`.
        Global configurations will remain intact.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IOTensor`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromKafka")):
      return KafkaIOTensor(
          subscription, servers, timeout, eof, conf, internal=True)

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
  # Dataset Conversions
  #=============================================================================

  def to_dataset(self):
    """Converts this `IOTensor` into a `tf.data.Dataset`.

    Example:

    ```python
    ```

    Args:

    Returns:
      A `tf.data.Dataset` with value obtained from this `IOTensor`.
    """
    class _IOTensorDataset(data_ops.BaseDataset):
      """_IOTensorDataset"""

      def __init__(self, dtype, shape, resource, function):
        start = 0
        stop = shape[0]
        capacity = 4096
        entry_start = list(range(start, stop, capacity))
        entry_stop = entry_start[1:] + [stop]
        dataset = data_ops.BaseDataset.from_tensor_slices((
            tf.constant(entry_start, tf.int64),
            tf.constant(entry_stop, tf.int64))).map(
                lambda start, stop: function(
                    resource, start, stop, 1, dtype=dtype)).apply(
                        tf.data.experimental.unbatch())
        self._dataset = dataset
        self._resource = resource
        self._function = function
        shape = shape[1:]
        super(_IOTensorDataset, self).__init__(
            self._dataset._variant_tensor, [dtype], [shape]) # pylint: disable=protected-access

    return _IOTensorDataset(
        self._dtype, self._shape, self._resource, self._function)

class IOIterableTensor(_IOBaseTensor):
  """IOIterableTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               dtype,
               shape,
               function,
               properties,
               internal=False):
    """Creates an `IOIterableTensor`. """
    self._function = function
    super(IOIterableTensor, self).__init__(
        dtype, shape, properties, internal=internal)

  #=============================================================================
  # Iterator
  #=============================================================================
  def __iter__(self):
    return self._function()

class AudioIOTensor(IOTensor):
  """AudioIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               internal=False):
    with tf.name_scope("AudioIOTensor") as scope:
      resource, dtypes, shapes, rate = core_ops.wav_indexable_init(
          filename, container=scope, shared_name=filename)
      dtype = tf.as_dtype(dtypes[0].numpy())
      shape = tf.TensorShape([
          None if dim < 0 else dim for dim in shapes[0].numpy() if dim != 0])
      properties = collections.OrderedDict({"rate": rate.numpy()})
      self._rate = rate.numpy()
      super(AudioIOTensor, self).__init__(
          dtype, shape,
          resource, core_ops.wav_indexable_get_item,
          properties,
          internal=internal)

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def rate(self):
    """The sampel `rate` of the audio stream"""
    return self._rate

class KafkaIOTensor(IOIterableTensor):
  """KafkaIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               subscription, servers, timeout, eof, conf,
               internal=False):
    with tf.name_scope("KafkaIOTensor") as scope:
      dtype = tf.string
      shape = tf.TensorShape([None])

      def function():
        """_iter_func"""
        metadata = []
        if servers is not None:
          metadata.append("bootstrap.servers=%s" % servers)
        if timeout is not None:
          metadata.append("conf.timeout=%d" % timeout)
        if eof is not None:
          metadata.append("conf.eof=%d" % (1 if eof else 0))
        if conf is not None:
          for e in conf:
            metadata.append(e)
        resource, _, _ = kafka_ops.kafka_iterable_init(
            subscription, metadata=metadata,
            container=scope, shared_name=subscription)
        capacity = 1
        while True:
          value = kafka_ops.kafka_iterable_next(resource, capacity=capacity)
          if tf.shape(value)[0].numpy() < capacity:
            return
          yield value

      super(KafkaIOTensor, self).__init__(
          dtype, shape, function, properties=None, internal=internal)
