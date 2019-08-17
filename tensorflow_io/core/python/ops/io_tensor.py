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

import sys
import collections
import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops

class _IOBaseTensor(object):
  """_IOBaseTensor"""

  def __init__(self,
               spec,
               properties,
               internal=False):
    if not internal:
      raise ValueError("IOTensor constructor is private; please use one "
                       "of the factory methods instead (e.g., "
                       "IOTensor.from_tensor())")
    self._spec = spec
    self._properties = collections.OrderedDict(
        {} if properties is None else properties)
    super(_IOBaseTensor, self).__init__()

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def spec(self):
    """The `TensorSpec` of values in this tensor."""
    return self._spec

  @property
  def dtype(self):
    """The `DType` of values in this tensor."""
    return tf.nest.map_structure(lambda e: e.dtype, self._spec)

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
    return tf.nest.map_structure(lambda e: e.shape, self._spec)

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
    return "<%s: spec=%s%s>" % (
        type(self).__name__, self.spec, props)


class IOTensor(_IOBaseTensor):
  """IOTensor

  An `IOTensor` is a tensor with data backed by IO operations. For example,
  an `AudioIOTensor` is a tensor with data from an audio file, a
  `KafkaIOTensor` is a tensor with data from reading the messages of a Kafka
  stream server.

  There are two types of `IOTensor`, a normal `IOTensor` which itself is
  indexable, or a degenerated `IOIterableTensor`  which only supports
  accessing the tensor iteratively.

  Since `IOTensor` is indexable, it support `__getitem__()` and
  `__len__()` methods in Python. In other words, it is a subclass of
  `collections.abc.Sequence`.

  Example:

  ```python
  >>> import tensorflow_io as tfio
  >>>
  >>> samples = tfio.IOTensor.from_audio("sample.wav")
  >>> print(samples[1000:1005])
  ... tf.Tensor(
  ... [[-3]
  ...  [-7]
  ...  [-6]
  ...  [-6]
  ...  [-5]], shape=(5, 1), dtype=int16)
  ```

  A `IOIterableTensor` is really a subclass of `collections.abc.Iterable`.
  It provides a `__iter__()` method that could be used (through `iter`
  indirectly) to access data in an iterative fashion.

  Example:

  ```python
  >>> import tensorflow_io as tfio
  >>>
  >>> kafka = tfio.IOTensor.from_kafka("test", eof=True)
  >>> for message in kafka:
  >>>   print(message)
  ... tf.Tensor(['D0'], shape=(1,), dtype=string)
  ... tf.Tensor(['D1'], shape=(1,), dtype=string)
  ... tf.Tensor(['D2'], shape=(1,), dtype=string)
  ... tf.Tensor(['D3'], shape=(1,), dtype=string)
  ... tf.Tensor(['D4'], shape=(1,), dtype=string)
  ```

  ### Indexable vs. Iterable

  While many IO formats are natually considered as iterable only, in most
  of the situations they could still be accessed by indexing through certain
  workaround. For example, a Kafka stream is not directly indexable yet the
  stream could be saved in memory or disk to allow indexing. Another example
  is the packet capture (PCAP) file in networking area. The packets inside
  a PCAP file is concatenated sequentially. Since each packets could have
  a variable length, the only way to access each packet is to read one
  packet at a time. If the PCAP file is huge (e.g., hundreds of GBs or even
  TBs), it may not be realistic (or necessarily) to save the index of every
  packet in memory. We could consider PCAP format as iterable only.

  As we could see the availability of memory size could be a factor to decide
  if a format is indexable or not. However, this factor could also be blurred
  as well in distributed computing. One common case is the file format that
  might be splittable where a file could be split into multiple chunks
  (without read the whole file) with no data overlapping in between those
  chunks. For example, a text file could be reliably split into multiple
  chunks with line feed (LF) as the boundary. Processing of chunks could then
  be distributed across a group of compute node to speed up (by reading small
  chunks into memory). From that standpoint, we could still consider splittable
  formats as indexable.

  For that reason our focus is `IOTensor` with convinience indexing and slicing
  through `__getitem__()` method.

  ### Lazy Read

  One useful feature of `IOTensor` is the lazy read. Data inside a file is not
  read into memory until needed. This could be convenient where only a small
  segment of the data is needed. For example, a WAV file could be as big as
  GBs but in many cases only several seconds of samples are used for training
  or inference purposes.

  While CPU memory is cheap nowadays, GPU memory is still considered as an
  expensive resource. It is also imperative to fit data in GPU memory for
  speed up purposes. From that perspective lazy read could be very helpful.

  ### Association of Meta Data

  While a file format could consist of mostly numeric data, in may situations
  the meta data is important as well. For example, in audio file format the
  sample rate is a number that is necessary for almost everything. Association
  of the sample rate with the sample of int16 Tensor is more helpful,
  especially in eager mode.

  Example:

  ```python
  >>> import tensorflow_io as tfio
  >>>
  >>> samples = tfio.IOTensor.from_audio("sample.wav")
  >>> print(samples.rate)
  ... 44100
  ```

  ### Nested Element Structure

  The concept of `IOTensor` is not limited to a Tensor of single data type.
  It supports nested element structure which could consists of many
  components and complex structures. The exposed API such as `shape()` or
  `dtype()` will display the shape and data type of an individual Tensor,
  or a nested structure of shape and data types for components of a
  composite Tensor.

  Example:

  ```python
  >>> import tensorflow_io as tfio
  >>>
  >>> samples = tfio.IOTensor.from_audio("sample.wav")
  >>> print(samples.shape)
  ... (22050, 2)
  >>> print(samples.dtype)
  ... <dtype: 'int32'>
  >>>
  >>> features = tfio.IOTensor.from_json("feature.json")
  >>> print(features.shape)
  ... (TensorShape([Dimension(2)]), TensorShape([Dimension(2)]))
  >>> print(features.dtype)
  ... (tf.float64, tf.int64)
  ```

  ### Access Columns of Tabular Data Formats

  May file formats such as Parquet or Json are considered as Tabular because
  they consists of columns in a table. With `IOTensor` it is possible to
  access individual columns through `__call__()`.

  Example:

  ```python
  >>> import tensorflow_io as tfio
  >>>
  >>> features = tfio.IOTensor.from_json("feature.json")
  >>> print(features.shape)
  ... (TensorShape([Dimension(2)]), TensorShape([Dimension(2)]))
  >>> print(features.dtype)
  ... (tf.float64, tf.int64)
  >>>
  >>> print(features("floatfeature").shape)
  ... (2,)
  >>> print(features("floatfeature").dtype)
  ... <dtype: 'float64'>
  ```

  ### Conversion from and to Tensor and Dataset

  When needed, `IOTensor` could be converted into a `Tensor` (through
  `to_tensor()`, or a `tf.data.Dataset` (through `to_dataset()`, to
  suppor operations that is only available through `Tensor` or
  `tf.data.Dataset`.

  Example:

  ```python
  >>> import tensorflow as tf
  >>> import tensorflow_io as tfio
  >>>
  >>> features = tfio.IOTensor.from_json("feature.json")
  >>>
  >>> features_tensor = features.to_tensor()
  >>> print(features_tensor())
  ... (<tf.Tensor: id=21, shape=(2,), dtype=float64, numpy=array([1.1, 2.1])>,
  ...  <tf.Tensor: id=22, shape=(2,), dtype=int64, numpy=array([2, 3])>)
  >>>
  >>> features_dataset = features.to_dataset()
  >>> print(features_dataset)
  ... <_IOTensorDataset shapes: ((), ()), types: (tf.float64, tf.int64)>
  >>>
  >>> dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
  ```

  """

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               spec,
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
      spec: The `TensorSpec` of the tensor.
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
    super(IOTensor, self).__init__(spec, properties, internal=internal)

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
    dtype = tf.nest.flatten(
        tf.nest.map_structure(lambda e: e.dtype, self.spec))
    shape = tf.nest.flatten(
        tf.nest.map_structure(lambda e: e.shape, self.spec))
    return tf.nest.pack_sequence_as(self.spec, self._function(
        self._resource,
        start, stop, step,
        dtype=dtype,
        shape=shape))

  def __len__(self):
    """Returns the total number of items of this IOTensor."""
    return tf.nest.flatten(
        tf.nest.map_structure(lambda e: e.shape, self.spec))[0][0]

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
  def from_json(cls,
                filename,
                **kwargs):
    """Creates an `IOTensor` from an json file.

    Args:
      filename: A string, the filename of an json file.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IOTensor`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromJSON")):
      return JSONIOTensor(filename, internal=True)

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
    class _IOTensorDataset(tf.compat.v2.data.Dataset):
      """_IOTensorDataset"""

      def __init__(self, spec, resource, function):
        start = 0
        stop = tf.nest.flatten(
            tf.nest.map_structure(lambda e: e.shape, spec))[0][0]
        capacity = 4096
        entry_start = list(range(start, stop, capacity))
        entry_stop = entry_start[1:] + [stop]

        dtype = tf.nest.flatten(
            tf.nest.map_structure(lambda e: e.dtype, spec))
        shape = tf.nest.flatten(
            tf.nest.map_structure(
                lambda e: tf.TensorShape(
                    [None]).concatenate(e.shape[1:]), spec))

        dataset = tf.compat.v2.data.Dataset.from_tensor_slices((
            tf.constant(entry_start, tf.int64),
            tf.constant(entry_stop, tf.int64)))
        dataset = dataset.map(
            lambda start, stop: function(
                resource, start, stop, 1, dtype=dtype, shape=shape))
        # Note: tf.data.Dataset consider tuple `(e, )` as one element
        # instead of a sequence. So next `unbatch()` will not work.
        # The tf.stack() below is necessary.
        if len(dtype) == 1:
          dataset = dataset.map(tf.stack)
        dataset = dataset.apply(tf.data.experimental.unbatch())
        self._dataset = dataset
        self._resource = resource
        self._function = function
        super(_IOTensorDataset, self).__init__(
            self._dataset._variant_tensor) # pylint: disable=protected-access

      def _inputs(self):
        return []

      @property
      def _element_structure(self):
        return self._dataset._element_structure # pylint: disable=protected-access

    return _IOTensorDataset(
        self.spec, self._resource, self._function)

class IOIterableTensor(_IOBaseTensor):
  """IOIterableTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               spec,
               function,
               properties,
               internal=False):
    """Creates an `IOIterableTensor`. """
    self._function = function
    super(IOIterableTensor, self).__init__(
        spec, properties, internal=internal)

  #=============================================================================
  # Iterator
  #=============================================================================
  def __iter__(self):
    dtype = tf.nest.flatten(
        tf.nest.map_structure(lambda e: e.dtype, self.spec))
    shape = tf.nest.flatten(
        tf.nest.map_structure(lambda e: e.shape, self.spec))
    resource = self._function["init"](self._function["data"])
    capacity = 1
    while True:
      value = self._function["next"](
          resource, capacity=capacity, dtype=dtype, shape=shape)
      if tf.shape(value[0])[0].numpy() < capacity:
        return
      yield tf.nest.pack_sequence_as(self.spec, value)


  #=============================================================================
  # Dataset Conversions
  #=============================================================================

  def to_dataset(self):
    """Converts this `IOIterableTensor` into a `tf.data.Dataset`.

    Example:

    ```python
    ```

    Args:

    Returns:
      A `tf.data.Dataset` with value obtained from this `IOIterableTensor`.
    """
    class _IOIterableTensorDataset(tf.compat.v2.data.Dataset):
      """_IOIterableTensorDataset"""

      def __init__(self, spec, function):
        func_init = function["init"]
        func_next = function["next"]
        func_data = function["data"]

        dtype = tf.nest.flatten(
            tf.nest.map_structure(lambda e: e.dtype, spec))
        shape = tf.nest.flatten(
            tf.nest.map_structure(
                lambda e: tf.TensorShape(
                    [None]).concatenate(e.shape[1:]), spec))

        resource = func_init(func_data)
        capacity = 4096
        dataset = tf.compat.v2.data.Dataset.range(0, sys.maxsize, capacity)
        dataset = dataset.map(
            lambda i: func_next(resource, capacity, dtype=dtype, shape=shape))
        dataset = dataset.apply(
            tf.data.experimental.take_while(
                lambda v: tf.greater(tf.shape(v)[0], 0)))
        # Note: tf.data.Dataset consider tuple `(e, )` as one element
        # instead of a sequence. So next `unbatch()` will not work.
        # The tf.stack() below is necessary.
        if len(dtype) == 1:
          dataset = dataset.map(tf.stack)
        dataset = dataset.apply(tf.data.experimental.unbatch())
        self._dataset = dataset
        self._resource = resource
        self._function = function
        super(_IOIterableTensorDataset, self).__init__(
            self._dataset._variant_tensor) # pylint: disable=protected-access

      def _inputs(self):
        return []

      @property
      def _element_structure(self):
        return self._dataset._element_structure # pylint: disable=protected-access

    return _IOIterableTensorDataset(
        self.spec, self._function)

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
          filename,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      shapes = [
          tf.TensorShape(
              [None if dim < 0 else dim for dim in e.numpy() if dim != 0]
          ) for e in tf.unstack(shapes)]
      dtypes = [tf.as_dtype(e.numpy()) for e in tf.unstack(dtypes)]
      spec = [tf.TensorSpec(shape, dtype) for (
          shape, dtype) in zip(shapes, dtypes)]
      if len(spec) == 1:
        spec = spec[0]
      else:
        spec = tuple(spec)
      properties = collections.OrderedDict({"rate": rate.numpy()})
      self._rate = rate.numpy()
      super(AudioIOTensor, self).__init__(
          spec, resource, core_ops.wav_indexable_get_item,
          properties,
          internal=internal)

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def rate(self):
    """The sampel `rate` of the audio stream"""
    return self._rate

class JSONIOTensor(IOTensor):
  """JSONIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               filename,
               columns=None,
               internal=False):
    with tf.name_scope("JSONIOTensor") as scope:
      metadata = []
      if columns is not None:
        metadata.extend(["column: "+column for column in columns])
      resource, dtypes, shapes, columns = core_ops.json_indexable_init(
          filename, metadata=metadata,
          container=scope,
          shared_name="%s/%s" % (filename, uuid.uuid4().hex))
      shapes = [
          tf.TensorShape(
              [None if dim < 0 else dim for dim in e.numpy() if dim != 0]
          ) for e in tf.unstack(shapes)]
      dtypes = [tf.as_dtype(e.numpy()) for e in tf.unstack(dtypes)]
      columns = [e.numpy() for e in tf.unstack(columns)]
      spec = [tf.TensorSpec(shape, dtype, column) for (
          shape, dtype, column) in zip(shapes, dtypes, columns)]
      if len(spec) == 1:
        spec = spec[0]
      else:
        spec = tuple(spec)
      self._filename = filename
      super(JSONIOTensor, self).__init__(
          spec, resource, core_ops.json_indexable_get_item,
          None,
          internal=internal)

  #=============================================================================
  # Accessors
  #=============================================================================

  def column(self, name):
    """The `TensorSpec` of column named `name`"""
    return next(e for e in tf.nest.flatten(self.spec) if e.name == name)

  def __call__(self, column):
    """Return a new JSONIOTensor with column named `column`"""
    return JSONIOTensor(self._filename, columns=[column], internal=True)

class KafkaIOTensor(IOIterableTensor):
  """KafkaIOTensor"""

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               subscription, servers, timeout, eof, conf,
               internal=False):
    with tf.name_scope("KafkaIOTensor") as scope:

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

      func_data = {"subscription": subscription, "metadata": metadata}
      def func_init(data):
        """func_init"""
        resource, _, _ = core_ops.kafka_iterable_init(
            data["subscription"], metadata=data["metadata"],
            container=scope,
            shared_name="%s/%s" % (subscription, uuid.uuid4().hex))
        return resource
      func_next = core_ops.kafka_iterable_next

      super(KafkaIOTensor, self).__init__(
          tf.TensorSpec(tf.TensorShape([None]), tf.string),
          {"init": func_init, "next": func_next, "data": func_data},
          properties=None, internal=internal)
