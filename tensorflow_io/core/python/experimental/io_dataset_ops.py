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
"""tensorflow_io.experimental.IODataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_io.core.python.ops import io_dataset
from tensorflow_io.core.python.experimental import libsvm_dataset_ops
from tensorflow_io.core.python.experimental import image_dataset_ops
from tensorflow_io.core.python.experimental import kinesis_dataset_ops
from tensorflow_io.core.python.experimental import pubsub_dataset_ops
from tensorflow_io.core.python.experimental import grpc_dataset_ops
from tensorflow_io.core.python.experimental import file_dataset_ops

class IODataset(io_dataset.IODataset):
  """IODataset"""

  #=============================================================================
  # Stream mode
  #=============================================================================

  @classmethod
  def stream(cls):
    """Obtain a non-repeatable StreamIODataset to be used.

    Returns:
      A class of `StreamIODataset`.
    """
    return StreamIODataset

  #=============================================================================
  # Factory Methods
  #=============================================================================

  @classmethod
  def from_libsvm(cls,
                  filename,
                  num_features,
                  dtype=None,
                  label_dtype=None,
                  compression_type='',
                  **kwargs):
    """Creates an `IODataset` from a libsvm file.

    Args:
      filename: A `tf.string` tensor containing one or more filenames.
      num_features: The number of features.
      dtype(Optional): The type of the output feature tensor.
        Default to tf.float32.
      label_dtype(Optional): The type of the output label tensor.
        Default to tf.int64.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IODataset`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromLibSVM")):
      return libsvm_dataset_ops.LibSVMIODataset(
          filename, num_features,
          dtype=dtype, label_dtype=label_dtype,
          compression_type=compression_type,
          internal=True, **kwargs)

  @classmethod
  def from_tiff(cls,
                filename,
                **kwargs):
    """Creates an `IODataset` from a TIFF file.

    Args:
      filename: A string, the filename of a TIFF file.
      name: A name prefix for the IOTensor (optional).

    Returns:
      A `IODataset`.

    """
    with tf.name_scope(kwargs.get("name", "IOFromTIFF")):
      return image_dataset_ops.TIFFIODataset(
          filename, internal=True)

  @classmethod
  def from_kinesis(cls,
                   stream,
                   shard="",
                   **kwargs):
    """Creates an `IODataset` from a Kinesis stream.

    Args:
      stream: A string, the stream name.
      shard: A string, the shard of kinesis.
      name: A name prefix for the IODataset (optional).

    Returns:
      A `IODataset`.
    """
    with tf.name_scope(kwargs.get("name", "IOFromKinesis")):
      return kinesis_dataset_ops.KinesisIODataset(
          stream, shard, internal=True)

  @classmethod
  def to_file(cls,
              dataset,
              filename,
              **kwargs):
    """Write dataset to a file.

    Args:
      dataset: A dataset whose content will be written to.
      filename: A string, the filename of the file to write to.
      name: A name prefix for the IODataset (optional).

    Returns:
      The number of records written.
    """
    with tf.name_scope(kwargs.get("name", "IOToText")):
      return file_dataset_ops.to_file(dataset, filename)

class StreamIODataset(tf.data.Dataset):
  """StreamIODataset"""

  @classmethod
  def from_prometheus_scrape(cls,
                             metric,
                             endpoint,
                             interval=None,
                             **kwargs):
    """Creates an `StreamIODataset` from a prometheus scrape endpoint.

    Args:
      metric: A string, the name of the metric to scrape.
      endpoint: A string, the address of prometheus scrape endpoint.
      interval: An integer, the time interval to scrape, by default 1s.
      name: A name prefix for the IODataset (optional).

    Returns:
      A `IODataset`.
    """
    with tf.name_scope(kwargs.get("name", "IOFromPrometheusScrape")):
      from tensorflow_io.core.python.ops import prometheus_dataset_ops # pylint: disable=import-outside-toplevel
      return prometheus_dataset_ops.PrometheusScrapeStreamIODataset(
          metric, endpoint, interval=interval, internal=True)

  @classmethod
  def from_pubsub(cls,
                  subscription,
                  endpoint=None,
                  timeout=10000,
                  **kwargs):
    """Creates an `StreamIODataset` from a pubsub endpoint.

    Args:
      subscription: A string, the subscription of the pubsub messages.
      endpoint: A string, the address of pubsub endpoint.
      timeout: An integer, the timeout of the pubsub pull.
      name: A name prefix for the IODataset (optional).

    Returns:
      A `IODataset`.
    """
    with tf.name_scope(kwargs.get("name", "IOFromPubSub")):
      return pubsub_dataset_ops.PubSubStreamIODataset(
          subscription, endpoint=endpoint, timeout=timeout, internal=True)

  @classmethod
  def from_grpc_numpy(cls, a, **kwargs):
    """Creates an `StreamIODataset` from numpy array through grpc.

    Args:
      a: A numpy array.
      name: A name prefix for the IODataset (optional).

    Returns:
      A `IODataset`.
    """
    with tf.name_scope(kwargs.get("name", "IOFromGRPC")):
      return grpc_dataset_ops.GRPCStreamIODataset.from_numpy(a, internal=True)
