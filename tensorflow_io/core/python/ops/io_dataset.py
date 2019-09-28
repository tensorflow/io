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
"""IODataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_io.core.python.ops import io_dataset_ops
from tensorflow_io.core.python.ops import kafka_dataset_ops

class IODataset(io_dataset_ops._IODataset):  # pylint: disable=protected-access
  """IODataset

  An `IODataset` is a subclass of `tf.data.Dataset` that is definitive with
  with data backed by IO operations. It is definitive so data should be both
  bounded and repeatable. In other words, the data will run out eventually
  (bounded) and a re-run of the `IODataset` will create an exact same sequence
  of data. `IODataset` could be passed to `tf.keras` for both training and
  inference purposes.

  Note while `IODataset` is a subclass of `tf.data.Dataset`, definitive
  does not necessarily apply to all `tf.data.Dataset`. A `tf.data.Dataset`
  could also be a `IOStreamDataset` where data generation may last forever
  and/or non-repeatable. `IOStreamDataset` is only safe to be passed to
  `tf.keras` for inference purposes.

  Examples of `IODataset` include `AudioIODataset` which is based on data
  from an audio (e.g., WAV) file. `KafkaIODataset` is also an `IODataset`
  which could be re-run multiple times and expect the same generated data.

  Example:

  ```python
  >>> import tensorflow as tf
  >>> import tensorflow_io as tfio
  >>>
  >>> audio = tfio.IODataset.from_audio("sample.wav")
  >>>
  ```

  """

  #=============================================================================
  # Factory Methods
  #=============================================================================

  @classmethod
  def from_audio(cls,
                 filename,
                 **kwargs):
    """Creates an `IODataset` from an audio file.

    The following audio file formats are supported:
    - WAV

    Args:
      filename: A string, the filename of an audio file.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IODataset`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromAudio")):
      raise NotImplementedError

  @classmethod
  def from_kafka(cls,
                 topic,
                 partition=0,
                 offset=0,
                 tail=-1,
                 **kwargs):
    """Creates an `IODataset` from kafka server with an offset range.

    Args:
      topic: A `tf.string` tensor containing topic subscription.
      partition: A `tf.int64` tensor containing the partition, by default 0.
      offset: A `tf.int64` tensor containing the start offset, by default 0.
      tail: A `tf.int64` tensor containing the end offset, by default -1.
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
      name: A name prefix for the IODataset (optional).

    Returns:
      A `IODataset`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromKafka")):
      return kafka_dataset_ops.KafkaIODataset(
          topic, partition=partition, offset=offset, tail=tail,
          servers=kwargs.get("servers", None),
          configuration=kwargs.get("configuration", None),
          internal=True)

class IOStreamDataset(io_dataset_ops._IOStreamDataset):  # pylint: disable=protected-access
  """IOStreamDataset

  An `IOStreamDataset` is a subclass of `tf.data.Dataset` that does not have
  to be definitive, with data backed by IO operations. The data generated from
  `IOStreamDataset` may run forever (unbounded). A re-run of `IOStreamDataset`
  could also have a completely different sequence of data (non-repeatable).
  While `IOStreamDataset` could be passed to `tf.keras`, is only suited for
  inference purposes. As a comparision, `IODataset` could be used for both
  training and inference purposes.

  Examples of `IOStreamDataset` include `KafkaIOStreamDataset` which is from
  a Kafka server, without a definitive beginning offset, or without an end
  offset.

  Example:

  ```python
  >>> import tensorflow as tf
  >>> import tensorflow_io as tfio
  >>>
  >>> kafka = tfio.IOStreamDataset.from_kafka("test-topic", offset=0)
  >>>
  ```

  """

  #=============================================================================
  # Factory Methods
  #=============================================================================

  @classmethod
  def from_kafka(cls,
                 topic,
                 partition=0,
                 offset=0,
                 **kwargs):
    """Creates an `IODataset` from kafka server with only a start offset.

    Args:
      topic: A `tf.string` tensor containing topic subscription.
      partition: A `tf.int64` tensor containing the partition, by default 0.
      offset: A `tf.int64` tensor containing the start offset, by default 0.
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
        Dataset configuration: there is one configuration available,
          `conf.timeout=milliseconds`: timeout value for Kafka Consumer to wait.
      name: A name prefix for the IODataset (optional).

    Returns:
      A `IOStreamDataset`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromKafka")):
      return kafka_dataset_ops.KafkaIOStreamDataset(
          topic, partition=partition, offset=offset,
          servers=kwargs.get("servers", None),
          configuration=kwargs.get("configuration", None),
          internal=True)
