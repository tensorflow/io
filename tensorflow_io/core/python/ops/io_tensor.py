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
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import audio_io_tensor_ops
from tensorflow_io.core.python.ops import json_io_tensor_ops

class IOTensor(io_tensor_ops._BaseIOTensor):
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
      return audio_io_tensor_ops.AudioIOTensor(filename, internal=True)

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
      return json_io_tensor_ops.JSONIOTensor(filename, internal=True)
