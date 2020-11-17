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
import enum
import tensorflow as tf
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

    class DataFormat(enum.Enum):
        """Data serialization format to use when reading from BigQuery."""

        AVRO = "AVRO"
        ARROW = "ARROW"

    class FieldMode(enum.Enum):
        """BigQuery column mode."""

        NULLABLE = "NULLABLE"
        REQUIRED = "REQUIRED"
        REPEATED = "REPEATED"

    def __init__(self):
        """Creates a BigQueryClient to start BigQuery read sessions."""
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
        data_format: DataFormat = DataFormat.AVRO,
    ):
        """Opens a session and returns a `BigQueryReadSession` object.

        Args:
            parent: String of the form projects/{project_id} indicating the project
                this ReadSession is associated with. This is the project that will be
                billed for usage.
            project_id: The assigned project ID of the project.
            table_id: The ID of the table in the dataset.
            dataset_id: The ID of the dataset in the project.
            selected_fields: This can be a list or a dict. If a list, it has
                names of the fields in the table that should be read. If a dict,
                it should be in a form like, i.e:
                { "field_a_name": {"mode": "repeated", output_type: dtypes.int64},
                "field_b_name": {"mode": "nullable", output_type: dtypes.string},
                ...
                "field_x_name": {"mode": "repeated", output_type: dtypes.string}
                }
                "mode" is BigQuery column attribute, it can be 'repeated', 'nullable' or 'required'.
                The output field order is unrelated to the order of fields in
                selected_fields. If "mode" not specified, defaults to "nullable".
                If "output_type" not specified, DT_STRING is implied for all Tensors.
            output_types: Types for the output tensor in the same sequence as
                selected_fields. This is only needed when selected_fields is a list,
                if selected_fields is a dictionary, this output_types information is
                included in selected_fields as described above.
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

        if isinstance(selected_fields, list):
            if not isinstance(output_types, list):
                raise ValueError(
                    "`output_types` must be a list if selected_fields is list"
                )
            if output_types and len(output_types) != len(selected_fields):
                raise ValueError(
                    "lengths of `output_types` must be a same as the "
                    "length of `selected_fields`"
                )
            if not output_types:
                output_types = [dtypes.string] * len(selected_fields)
            # Repeated field is not supported if selected_fields is list
            selected_fields_repeated = [False] * len(selected_fields)

        elif isinstance(selected_fields, dict):
            _selected_fields = []
            selected_fields_repeated = []
            output_types = []
            for field in selected_fields:
                _selected_fields.append(field)
                mode = selected_fields[field].get("mode", self.FieldMode.NULLABLE)
                if mode == self.FieldMode.REPEATED:
                    selected_fields_repeated.append(True)
                elif mode == self.FieldMode.NULLABLE or mode == self.FieldMode.REQUIRED:
                    selected_fields_repeated.append(False)
                else:
                    raise ValueError(
                        "mode needs be BigQueryClient.FieldMode.NULLABLE, FieldMode.REQUIRED or FieldMode.REPEATED"
                    )
                output_types.append(
                    selected_fields[field].get("output_type", dtypes.string)
                )
            selected_fields = _selected_fields
        else:
            raise ValueError("`selected_fields` must be a list or dict.")

        (streams, schema) = core_ops.io_big_query_read_session(
            client=self._client_resource,
            parent=parent,
            project_id=project_id,
            table_id=table_id,
            dataset_id=dataset_id,
            requested_streams=requested_streams,
            data_format=data_format.value,
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
            selected_fields_repeated,
            output_types,
            row_restriction,
            requested_streams,
            data_format,
            streams,
            schema,
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
        selected_fields_repeated,
        output_types,
        row_restriction,
        requested_streams,
        data_format,
        streams,
        schema,
        client_resource,
    ):
        self._parent = parent
        self._project_id = project_id
        self._table_id = table_id
        self._dataset_id = dataset_id
        self._selected_fields = selected_fields
        self._selected_fields_repeated = selected_fields_repeated
        self._output_types = output_types
        self._row_restriction = row_restriction
        self._requested_streams = requested_streams
        self._data_format = data_format
        self._streams = streams
        self._schema = schema
        self._client_resource = client_resource

    def get_streams(self):
        """Returns Tensor with stream names for reading data from BigQuery.

        Returns:
            Tensor with stream names.
        """
        return self._streams

    def read_rows(self, stream, offset=0):
        """Retrieves rows (including values) from the BigQuery service.

        Args:
            stream: name of the stream to read from.
            offset: Position in the stream.

        Returns:
            A `tf.data.Dataset` returning the row keys and the cell contents.

        Raises:
            ValueError: If the configured probability is unexpected.
        """
        return _BigQueryDataset(
            self._client_resource,
            self._selected_fields,
            self._selected_fields_repeated,
            self._output_types,
            self._schema,
            self._data_format,
            stream,
            offset,
        )

    def parallel_read_rows(
        self, cycle_length=None, sloppy=False, block_length=1, num_parallel_calls=None,
    ):
        """Retrieves rows from the BigQuery service in parallel streams.

        ```
        bq_client = BigQueryClient()
        bq_read_session = bq_client.read_session(...)
        ds1 = bq_read_session.parallel_read_rows(...)
        ```
        Args:
            cycle_length: number of streams to process in parallel. If not specified, it
                is defaulted to the number of streams in the read session.
            sloppy: If false, elements are produced in deterministic order. If true,
                the implementation is allowed, for the sake of expediency, to produce
                elements in a non-deterministic order.
                When reading from multiple BigQuery streams, setting sloppy=True usually
                yields a better performance.
            block_length: The number of consecutive elements to pull from a session stream
                before advancing to the next one.
            num_parallel_calls: Number of threads to use for processing input streams.
                If the value `tf.data.experimental.AUTOTUNE` is used, then the number of
                parallel calls is set dynamically based on available CPU.
                Defaulted to the number of streams in the read session.

        Returns:
            A `tf.data.Dataset` returning the row keys and the cell contents.

        Raises:
            ValueError: If the configured probability is unexpected.
        """
        streams_ds = dataset_ops.Dataset.from_tensor_slices(self._streams)
        streams_count = tf.cast(tf.size(self._streams), dtype=tf.int64)
        if cycle_length is None:
            cycle_length = streams_count
        if num_parallel_calls is None:
            num_parallel_calls = streams_count

        return streams_ds.interleave(
            map_func=self.read_rows,
            cycle_length=cycle_length,
            block_length=block_length,
            num_parallel_calls=num_parallel_calls,
            deterministic=not (sloppy),
        )


class _BigQueryDataset(dataset_ops.DatasetSource):
    """_BigQueryDataset represents a dataset that retrieves keys and values."""

    def __init__(
        self,
        client_resource,
        selected_fields,
        selected_fields_repeated,
        output_types,
        schema,
        data_format,
        stream,
        offset,
    ):
        # selected_fields and corresponding output_types have to be sorted because
        # of b/141251314
        sorted_fields_with_types = sorted(
            zip(selected_fields, selected_fields_repeated, output_types),
            key=itemgetter(0),
        )
        selected_fields, selected_fields_repeated, output_types = list(
            zip(*sorted_fields_with_types)
        )
        selected_fields = list(selected_fields)
        selected_fields_repeated = list(selected_fields_repeated)
        output_types = list(output_types)

        tensor_shapes = list(
            [None,] if repeated else [] for repeated in selected_fields_repeated
        )

        self._element_spec = collections.OrderedDict(
            zip(
                selected_fields,
                (
                    tensor_spec.TensorSpec(shape, dtype)
                    for (shape, dtype) in zip(tensor_shapes, output_types)
                ),
            )
        )

        variant_tensor = core_ops.io_big_query_dataset(
            client=client_resource,
            selected_fields=selected_fields,
            output_types=output_types,
            schema=schema,
            data_format=data_format.value,
            stream=stream,
            offset=offset,
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
