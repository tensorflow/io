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

import tensorflow as tf

from tensorflow_io.core.python.ops import io_dataset
from tensorflow_io.core.python.experimental import libsvm_dataset_ops
from tensorflow_io.core.python.experimental import image_dataset_ops
from tensorflow_io.core.python.experimental import kinesis_dataset_ops
from tensorflow_io.core.python.experimental import pubsub_dataset_ops
from tensorflow_io.core.python.experimental import grpc_dataset_ops
from tensorflow_io.core.python.experimental import file_dataset_ops
from tensorflow_io.core.python.experimental import numpy_dataset_ops
from tensorflow_io.core.python.experimental import sql_dataset_ops
from tensorflow_io.core.python.experimental import video_dataset_ops


class IODataset(io_dataset.IODataset):
    """IODataset"""

    # =============================================================================
    # Stream mode
    # =============================================================================

    @classmethod
    def stream(cls):
        """Obtain a non-repeatable StreamIODataset to be used.

        Returns:
          A class of `StreamIODataset`.
        """
        return StreamIODataset

    # =============================================================================
    # Factory Methods
    # =============================================================================

    @classmethod
    def from_libsvm(
        cls,
        filename,
        num_features,
        dtype=None,
        label_dtype=None,
        compression_type="",
        **kwargs
    ):
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
                filename,
                num_features,
                dtype=dtype,
                label_dtype=label_dtype,
                compression_type=compression_type,
                internal=True,
                **kwargs
            )

    @classmethod
    def from_tiff(cls, filename, **kwargs):
        """Creates an `IODataset` from a TIFF file.

        Args:
          filename: A string, the filename of a TIFF file.
          name: A name prefix for the IOTensor (optional).

        Returns:
          A `IODataset`.
        """
        with tf.name_scope(kwargs.get("name", "IOFromTIFF")):
            return image_dataset_ops.TIFFIODataset(filename, internal=True)

    @classmethod
    def from_kinesis(cls, stream, shard="", **kwargs):
        """Creates an `IODataset` from a Kinesis stream.

        Args:
          stream: A string, the stream name.
          shard: A string, the shard of kinesis.
          name: A name prefix for the IODataset (optional).

        Returns:
          A `IODataset`.
        """
        with tf.name_scope(kwargs.get("name", "IOFromKinesis")):
            return kinesis_dataset_ops.KinesisIODataset(stream, shard, internal=True)

    @classmethod
    def from_numpy(cls, a, **kwargs):
        """Creates an `IODataset` from Numpy arrays.

        The `from_numpy` allows user to create a Dataset from a dict,
        tuple, or individual element of numpy array_like.
        The `Dataset` created through `from_numpy` has the same dtypes
        as the input elements of array_like. The shapes of the `Dataset`
        is similar to the input elements of array_like, except that the
        first dimensions of the shapes are set to None. The reason is
        that first dimensions of the iterated output which may not be
        dividable to the total number of elements.

        For example:
        ```
        import numpy as np
        import tensorflow as tf
        import tensorflow_io as tfio
        a = (np.asarray([[0., 1.], [2., 3.], [4., 5.], [6., 7.], [8., 9.]]),
             np.asarray([[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]]))
        d = tfio.experimental.IODataset.from_numpy(a).batch(2)
        for i in d:
          print(i.numpy())
        # numbers of elements = [2, 2, 1] <= (5 / 2)
        #
        # ([[0., 1.], [2., 3.]], [[10, 11], [12, 13]]) # <= batch index 0
        # ([[4., 5.], [6., 7.]], [[14, 15], [16, 17]]) # <= batch index 1
        # ([[8., 9.]],           [[18, 19]])           # <= batch index 2
        ```
        Args:
          a: dict, tuple, or array_like
            numpy array if the input type is array_like;
            dict or tuple of numpy arrays if the input type is dict or tuple.
          name: A name prefix for the IOTensor (optional).

        Returns:
          A `IODataset` with the same dtypes as in array_like specified
            in `a`.

        """
        with tf.name_scope(kwargs.get("name", "IOFromNumpy")):
            return numpy_dataset_ops.NumpyIODataset(a, internal=True)

    @classmethod
    def from_numpy_file(cls, filename, spec=None, **kwargs):
        """Creates an `IODataset` from a Numpy file.

        The `from_numpy_file` allows user to create a Dataset from
        a numpy file (npy or npz). The `Dataset` created through
        `from_numpy_file` has the same dtypes as the elements in numpy
        file. The shapes of the Dataset is similar to the elements of
        numpy file, except the first dimensions of the shapes are set
        to None. The reason is that first dimensions of the iterated
        output which may not be dividable to the total number of elements.
        In case numpy file consists of unnamed elements, a tuple of numpy
        arrays are returned, otherwise a dict is returned for named
        elements.
        ```
        Args:
          filename: filename of numpy file (npy or npz).
          spec: A tuple of tf.TensorSpec or dtype, or a dict of
            `name:tf.TensorSpec` or `name:dtype` pairs to specify the dtypes
            in each element of the numpy file. In eager mode spec is automatically
            probed. In graph spec must be provided. If a tuple is provided for
            spec then it is assumed that numpy file consists of `arr_0`, `arr_2`...
            If a dict is provided then numpy file should consists of named
            elements.
          name: A name prefix for the IOTensor (optional).

        Returns:
          A `IODataset` with the same dtypes as of the array_like in numpy
            file (npy or npz).
        """
        with tf.name_scope(kwargs.get("name", "IOFromNumpyFile")):
            return numpy_dataset_ops.NumpyFileIODataset(
                filename, spec=spec, internal=True
            )

    @classmethod
    def from_prometheus(cls, query, length, offset=None, endpoint=None, spec=None):
        """Creates an `GraphIODataset` from a prometheus endpoint.

        Args:
          query: A string, the query string for prometheus.
          length: An integer, the length of the query (in seconds).
          offset: An integer, the a millisecond-precision timestamp, by default
            the time when graph runs.
          endpoint: A string, the server address of prometheus, by default
            `http://localhost:9090`.
          spec: A structured tf.TensorSpec of the dataset.
            The format should be {"job": {"instance": {"name": tf.TensorSpec}}}.
            In graph mode, spec is needed. In eager mode,
            spec is probed automatically.
          name: A name prefix for the IODataset (optional).

        Returns:
          A `IODataset`.
        """
        from tensorflow_io.core.python.ops import (  # pylint: disable=import-outside-toplevel
            prometheus_dataset_ops,
        )

        return prometheus_dataset_ops.PrometheusIODataset(
            query, length, offset=offset, endpoint=endpoint, spec=spec
        )

    @classmethod
    def from_sql(cls, query, endpoint=None, spec=None):
        """Creates an `GraphIODataset` from a postgresql server endpoint.

        Args:
          query: A string, the sql query string.
          endpoint: A string, the server address of postgresql server.
          spec: A structured (tuple) tf.TensorSpec of the dataset.
            In graph mode, spec is needed. In eager mode,
            spec is probed automatically.
          name: A name prefix for the IODataset (optional).

        Returns:
          A `IODataset`.
        """
        return sql_dataset_ops.SQLIODataset(query, endpoint=endpoint, spec=spec)

    @classmethod
    def from_video(cls, filename):
        """Creates an `GraphIODataset` from a video file.

          Args:
            filename: A string, the sql query string.
            name: A name prefix for the IODataset (optional).

          Returns:
            A `IODataset`.
        """
        return video_dataset_ops.VideoIODataset(filename)

    @classmethod
    def to_file(cls, dataset, filename, **kwargs):
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
    def from_video_capture(cls, device, **kwargs):
        """Creates an `StreamIODataset` from video capture device.

        Args:
          device: A string, the name of the device.
          name: A name prefix for the IODataset (optional).

        Returns:
          A `IODataset`.
        """
        with tf.name_scope(kwargs.get("name", "IOFromVideoCapture")):
            return video_dataset_ops.VideoCaptureIODataset(device, internal=True)

    @classmethod
    def from_prometheus_scrape(cls, metric, endpoint, interval=None, **kwargs):
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
            from tensorflow_io.core.python.ops import (  # pylint: disable=import-outside-toplevel
                prometheus_dataset_ops,
            )

            return prometheus_dataset_ops.PrometheusScrapeStreamIODataset(
                metric, endpoint, interval=interval, internal=True
            )

    @classmethod
    def from_pubsub(cls, subscription, endpoint=None, timeout=10000, **kwargs):
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
                subscription, endpoint=endpoint, timeout=timeout, internal=True
            )

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
