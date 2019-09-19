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
from tensorflow_io.core.python.ops import io_tensor_ops
from tensorflow_io.core.python.ops import audio_io_tensor_ops
from tensorflow_io.core.python.ops import json_io_tensor_ops
from tensorflow_io.core.python.ops import hdf5_io_tensor_ops
from tensorflow_io.core.python.ops import kafka_io_tensor_ops
from tensorflow_io.core.python.ops import lmdb_io_tensor_ops
from tensorflow_io.core.python.ops import prometheus_io_tensor_ops
from tensorflow_io.core.python.ops import feather_io_tensor_ops
from tensorflow_io.core.python.ops import csv_io_tensor_ops
from tensorflow_io.core.python.ops import avro_io_tensor_ops

class IOTensor(io_tensor_ops._IOTensor):  # pylint: disable=protected-access
  """IOTensor

  An `IOTensor` is a tensor with data backed by IO operations. For example,
  an `AudioIOTensor` is a tensor with data from an audio file, a
  `KafkaIOTensor` is a tensor with data from reading the messages of a Kafka
  stream server.

  The `IOTensor` is indexable, supporting `__getitem__()` and
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

  As we could see the, availability of memory size could be a factor to decide
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
  >>> print(features.shape("floatfeature"))
  ... (2,)
  >>> print(features.dtype("floatfeature"))
  ... <dtype: 'float64'>
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
      return io_tensor_ops.TensorIOTensor(tensor, internal=True)

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
      return json_io_tensor_ops.JSONIOTensor(
          filename, mode=kwargs.get('mode', None), internal=True)

  @classmethod
  def from_kafka(cls,
                 subscription,
                 **kwargs):
    """Creates an `IOTensor` from a Kafka stream.

    Args:
      subscription: A `tf.string` tensor containing subscription,
        in the format of [topic:partition:offset:length],
        by default length is -1 for unlimited.
      servers: An optional list of bootstrap servers, by default
         `localhost:9092`.
      configuration: An optional `tf.string` tensor containing
        configurations in [Key=Value] format. There are three
        types of configurations:
        Global configuration: please refer to 'Global configuration properties'
          in librdkafka doc. Examples include
          ["enable.auto.commit=false", "heartbeat.interval.ms=2000"]
        Topic configuration: please refer to 'Topic configuration properties'
          in librdkafka doc. Note all topic configurations should be
          prefixed with `configuration.topic.`. Examples include
          ["conf.topic.auto.offset.reset=earliest"]
        Dataset configuration: there are two configurations available,
          `conf.eof=0|1`: if True, the KafkaDaset will stop on EOF (default).
          `conf.timeout=milliseconds`: timeout value for Kafka Consumer to wait.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IOTensor`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromKafka")):
      return kafka_io_tensor_ops.KafkaIOTensor(
          subscription,
          servers=kwargs.get("servers", None),
          configuration=kwargs.get("configuration", None),
          internal=True)

  @classmethod
  def from_prometheus(cls,
                      query,
                      **kwargs):
    """Creates an `IOTensor` from a prometheus query.

    Args:
      query: A string, the query string for prometheus.
      endpoint: A string, the server address of prometheus, by default
       `http://localhost:9090`.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A (`IOTensor`, `IOTensor`) tuple that represents `timestamp`
        and `value`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromPrometheus")):
      return prometheus_io_tensor_ops.PrometheusIOTensor(
          query, endpoint=kwargs.get("endpoint", None), internal=True)

  @classmethod
  def from_feather(cls,
                   filename,
                   **kwargs):
    """Creates an `IOTensor` from an feather file.

    Args:
      filename: A string, the filename of an feather file.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IOTensor`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromFeather")):
      return feather_io_tensor_ops.FeatherIOTensor(filename, internal=True)

  @classmethod
  def from_lmdb(cls,
                filename,
                **kwargs):
    """Creates an `IOTensor` from a LMDB key/value store.

    Args:
      filename: A string, the filename of a LMDB file.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IOTensor`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromLMDB")):
      return lmdb_io_tensor_ops.LMDBIOTensor(filename, internal=True)

  @classmethod
  def from_hdf5(cls,
                filename,
                **kwargs):
    """Creates an `IOTensor` from an hdf5 file.

    Args:
      filename: A string, the filename of an hdf5 file.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IOTensor`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromHDF5")):
      return hdf5_io_tensor_ops.HDF5IOTensor(filename, internal=True)

  @classmethod
  def from_csv(cls,
               filename,
               **kwargs):
    """Creates an `IOTensor` from an csv file.

    Args:
      filename: A string, the filename of an csv file.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IOTensor`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromCSV")):
      return csv_io_tensor_ops.CSVIOTensor(filename, internal=True)

  @classmethod
  def from_avro(cls,
                filename,
                schema,
                **kwargs):
    """Creates an `IOTensor` from an avro file.

    Args:
      filename: A string, the filename of an avro file.
      schema: A string, the schema of an avro file.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IOTensor`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromAvro")):
      return avro_io_tensor_ops.AvroIOTensor(
          filename, schema, internal=True, **kwargs)
