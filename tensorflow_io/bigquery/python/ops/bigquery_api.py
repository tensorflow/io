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
"""The Python API for TensorFlow's Cloud BigQuery integration.

TensorFlow has support for reading from Cloud BigQuery. To use
TensorFlow + Cloud BigQuery integration, first create a BigQueryClient to
configure your connection to Cloud BigQuery, and then create a
BigQueryReadSession object to allow you to create numerous `tf.data.Dataset`s
to read data the underlying Cloud BigQuery dataset.

For background on Cloud BigQuery, see: https://cloud.google.com/bigquery .
"""


import collections
from operator import itemgetter

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import deprecation
from tensorflow_io.core.python.ops import core_ops


class BigQueryClient:
    """BigQueryClient is the entrypoint for interacting with Cloud BigQuery in TF.

  BigQueryClient encapsulates a connection to Cloud BigQuery, and exposes the
  `readSession` method to initiate a BigQuery read session.
  """

    def __init__(self):
        """Creates a BigQueryClient to start BigQuery read sessions.

    """
        self._client_resource = core_ops.io_big_query_client()

    def read_session(
        self,
        parent,
        project_id,
        table_id,
        dataset_id,
        selected_fields,
        output_types=None,
        row_restriction="",
        requested_streams=1,
    ):
        """Opens a session and returns a `BigQueryReadSession` object.

    Args:
      parent: String of the form projects/{project_id} indicating the project
        this ReadSession is associated with. This is the project that will be
        billed for usage.
      project_id: The assigned project ID of the project.
      table_id: The ID of the table in the dataset.
      dataset_id: The ID of the dataset in the project.
      selected_fields: Names of the fields in the table that should be read.
        The output field order is unrelated to the order of fields in
        selected_fields.
      output_types: Types for the output tensor in the same sequence as
        selected_fields.
        If not specified, DT_STRING is implied for all Tensors.
      row_restriction: Optional. SQL text filtering statement, similar to a
        WHERE clause in a query.
      requested_streams: Initial number of streams. If unset or 0, we will
        provide a value of streams so as to produce reasonable throughput.
        Must be non-negative. The number of streams may be lower than the
        requested number, depending on the amount parallelism that is reasonable
        for the table and the maximum amount of parallelism allowed by the
        system.

    Returns:
      A `BigQueryReadSession` Python object representing the
      operations available on the table.
    """
        if not isinstance(parent, str):
            raise ValueError("`parent` must be a string")
        if not parent:
            raise ValueError("`parent` must be a set")

        if not isinstance(project_id, str):
            raise ValueError("`project_id` must be a string")
        if not project_id:
            raise ValueError("`project_id` must be a set")

        if not isinstance(table_id, str):
            raise ValueError("`table_id` must be a string")
        if not table_id:
            raise ValueError("`table_id` must be a set")

        if not isinstance(dataset_id, str):
            raise ValueError("`dataset_id` must be a string")
        if not dataset_id:
            raise ValueError("`dataset_id` must be a set")

        if not isinstance(selected_fields, list):
            raise ValueError("`selected_fields` must be a list")
        if not selected_fields:
            raise ValueError("`selected_fields` must be a set")

        if not isinstance(output_types, list):
            raise ValueError("`output_types` must be a list")
        if output_types and len(output_types) != len(selected_fields):
            raise ValueError(
                "lengths of `output_types` must be a same as the "
                "length of `selected_fields`"
            )

        if not output_types:
            output_types = [dtypes.string] * len(selected_fields)

        (streams, avro_schema) = core_ops.io_big_query_read_session(
            client=self._client_resource,
            parent=parent,
            project_id=project_id,
            table_id=table_id,
            dataset_id=dataset_id,
            requested_streams=requested_streams,
            selected_fields=selected_fields,
            output_types=output_types,
            row_restriction=row_restriction,
        )
        return BigQueryReadSession(
            parent,
            project_id,
            table_id,
            dataset_id,
            selected_fields,
            output_types,
            row_restriction,
            requested_streams,
            streams,
            avro_schema,
            self._client_resource,
        )


class BigQueryReadSession:
    """Entry point for reading data from Cloud BigQuery."""

    def __init__(
        self,
        parent,
        project_id,
        table_id,
        dataset_id,
        selected_fields,
        output_types,
        row_restriction,
        requested_streams,
        streams,
        avro_schema,
        client_resource,
    ):
        self._parent = parent
        self._project_id = project_id
        self._table_id = table_id
        self._dataset_id = dataset_id
        self._selected_fields = selected_fields
        self._output_types = output_types
        self._row_restriction = row_restriction
        self._requested_streams = requested_streams
        self._streams = streams
        self._avro_schema = avro_schema
        self._client_resource = client_resource

    def get_streams(self):
        """Returns Tensor with stream names for reading data from BigQuery.

    Returns:
      Tensor with stream names.
    """
        return self._streams

    def read_rows(self, stream):
        """Retrieves rows (including values) from the BigQuery service.

    Args:
      stream: name of the stream to read from.

    Returns:
      A `tf.data.Dataset` returning the row keys and the cell contents.

    Raises:
      ValueError: If the configured probability is unexpected.
    """
        return _BigQueryDataset(
            self._client_resource,
            self._selected_fields,
            self._output_types,
            self._avro_schema,
            stream,
        )

    @deprecation.deprecated_args(
        None,
        "If sloppy execution is desired,"
        "use `tf.data.Options.experimental_deterministic`.",
        "sloppy",
    )
    def parallel_read_rows(
        self, cycle_length=None, sloppy=False, block_length=1, num_parallel_calls=None
    ):
        """Retrieves rows from the BigQuery service in parallel streams.

    ```
    bq_client = BigQueryClient()
    bq_read_session = bq_client.read_session(...)
    ds1 = bq_read_session.parallel_read_rows(...)
    ```
    Args:
      cycle_length: number of threads to run in parallel. If not specified, it
        is defaulted to the number of streams in a read session.
      sloppy: If false, elements are produced in deterministic order. If true,
        the implementation is allowed, for the sake of expediency, to produce
        elements in a non-deterministic order. Otherwise, whether the order is
        deterministic or non-deterministic depends on the
        `tf.data.Options.experimental_deterministic` value.
      block_length: The number of consecutive elements to pull from an input
        `Dataset` before advancing to the next input `Dataset`.
      block_length: The number of consecutive elements to pull from an input
        `Dataset` before advancing to the next input `Dataset`.
      num_parallel_calls: If specified, the implementation creates a threadpool,
        which is used to fetch inputs from cycle elements asynchronously and in
        parallel. The default behavior is to fetch inputs from cycle elements
        synchronously with no parallelism.
        If the value `tf.data.experimental.AUTOTUNE` is used, then the number of
        parallel calls is set dynamically based on available CPU.

    Returns:
      A `tf.data.Dataset` returning the row keys and the cell contents.

    Raises:
      ValueError: If the configured probability is unexpected.

    """
        if cycle_length is None:
            cycle_length = self._requested_streams
        streams_ds = dataset_ops.Dataset.from_tensor_slices(self._streams)
        option = streams_ds.options()
        if sloppy is True:
            option.experimental_deterministic = False
            streams_ds = streams_ds.with_options(option)
        elif sloppy is False:
            option.experimental_deterministic = True
            streams_ds = streams_ds.with_options(option)

        return streams_ds.interleave(
            map_func=self.read_rows,
            cycle_length=cycle_length,
            block_length=block_length,
            num_parallel_calls=num_parallel_calls,
        )


class _BigQueryDataset(dataset_ops.DatasetSource):
    """_BigQueryDataset represents a dataset that retrieves keys and values."""

    def __init__(
        self, client_resource, selected_fields, output_types, avro_schema, stream
    ):

        # selected_fields and corresponding output_types have to be sorted because
        # of b/141251314
        sorted_fields_with_types = sorted(
            zip(selected_fields, output_types), key=itemgetter(0)
        )
        selected_fields, output_types = list(zip(*sorted_fields_with_types))
        selected_fields = list(selected_fields)
        output_types = list(output_types)

        self._element_spec = collections.OrderedDict(
            zip(
                selected_fields,
                (tensor_spec.TensorSpec([], dtype) for dtype in output_types),
            )
        )

        variant_tensor = core_ops.io_big_query_dataset(
            client=client_resource,
            selected_fields=selected_fields,
            output_types=output_types,
            avro_schema=avro_schema,
            stream=stream,
        )
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec


class BigQueryTestClient(BigQueryClient):
    """BigQueryTestClient is the entrypoint for interacting with Fake Cloud BigQuery service."""

    # pylint: disable=super-init-not-called
    def __init__(self, fake_server_address):
        """Creates a BigQueryTestClient to start BigQuery read sessions.

    Args:
      fake_server_address: url for service faking Cloud BigQuery Storage API.
    """

        self._client_resource = core_ops.io_big_query_test_client(fake_server_address)
