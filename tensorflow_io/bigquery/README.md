# Google BigQuery

[BigQuery](https://cloud.google.com/bigquery/) is a serverless, highly-scalable,
and cost-effective cloud data warehouse with an in-memory BI Engine and machine
learning built in.

BigQuery connector relies on [BigQuery Storage API](https://cloud.google.com/bigquery/docs/reference/storage/).
that provides fast access to BigQuery managed storage by using an rpc-based
protocol.

## Prerequisites

In order to use BigQuery connector, you need to make sure that Google Cloud SDK
is propertly configured and that you have BigQuery Storage API enabled.
Depending on environment you are using some prerequisites might be already met.

1. [Select or create a GCP project.](https://console.cloud.google.com/cloud-resource-manager)
2. [Install and initialize the Cloud SDK.](https://cloud.google.com/sdk/docs/)
3. [Setup Authentication.](https://cloud.google.com/docs/authentication/#service_accounts)
If you choose to use [service account](https://cloud.google.com/docs/authentication/production)
authentication, please make sure that GOOGLE_APPLICATION_CREDENTIALS
environment variable is initialized with a path pointing to JSON file that
contains your service account key.
4. [Enable BigQuery Storage API.](https://cloud.google.com/bigquery/docs/reference/storage/#enabling_the_api)
5. If you see some errors related to roots.pem file in logs, you can solve it via either of the following approaches:

* copy the [gRPC `roots.pem` file][grpcPem] to
  `/usr/share/grpc/roots.pem` on your local machine, which is the default
  location where gRPC will look for this file
* export the environment variable `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` to point to
  the full path of the gRPC `roots.pem` file on your file system if it's in a
  different location

[grpcPem]: https://github.com/grpc/grpc/blob/master/etc/roots.pem

## Sample Use

BigQuery connector mostly follows [BigQuery Storage API flow](https://cloud.google.com/bigquery/docs/reference/storage/#basic_api_flow),
but hides complexity associated with decoding serialized data rows into Tensors.

1. Create a `BigQueryClient` client.
2. Use the `BigQueryClient` to create `BigQueryReadSession` object corresponding
    to a read session. A read session divides the contents of a BigQuery table
    into one or more streams, which can then be used to read data from the
    table.
3. Call parallel_read_rows on `BigQueryReadSession` object to read from multiple
    BigQuery streams in parallel.

The following example illustrates how to read particular columns from public
BigQuery dataset.

```python
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow_io.bigquery import BigQueryClient
from tensorflow_io.bigquery import BigQueryReadSession

GCP_PROJECT_ID = '<FILL_ME_IN>'
DATASET_GCP_PROJECT_ID = "bigquery-public-data"
DATASET_ID = "samples"
TABLE_ID = "wikipedia"

def main():
  ops.enable_eager_execution()
  client = BigQueryClient()
  read_session = client.read_session(
      "projects/" + GCP_PROJECT_ID,
      DATASET_GCP_PROJECT_ID, TABLE_ID, DATASET_ID,
      ["title",
       "id",
       "num_characters",
       "language",
       "timestamp",
       "wp_namespace",
       "contributor_username"],
      [dtypes.string,
       dtypes.int64,
       dtypes.int64,
       dtypes.string,
       dtypes.int64,
       dtypes.int64,
       dtypes.string],
      requested_streams=2,
      row_restriction="num_characters > 1000",
      data_format=BigQueryClient.DataFormat.AVRO)
  dataset = read_session.parallel_read_rows()

  row_index = 0
  for row in dataset.prefetch(10):
    print("row %d: %s" % (row_index, row))
    row_index += 1

if __name__ == '__main__':
  app.run(main)

```

It also supports reading BigQuery column with repeated mode (each field contains array of values with primitive type: Integer, Float, Boolean, String, but RECORD is not supported). In this case, selected_fields needs be a dictionary in a
form like

```python
  { "field_a_name": {"mode": BigQueryClient.FieldMode.REPEATED, output_type: dtypes.int64},
    "field_b_name": {"mode": BigQueryClient.FieldMode.NULLABLE, output_type: dtypes.string},
    ...
    "field_x_name": {"mode": BigQueryClient.FieldMode.REQUIRED, output_type: dtypes.string}
  }
```
"mode" is BigQuery column attribute concept, it can be 'repeated', 'nullable' or 'required' (enum BigQueryClient.FieldMode.REPEATED, NULLABLE, REQUIRED).The output field order is unrelated to the order of fields in
selected_fields. If "mode" not specified, defaults to "nullable". If "output_type" not specified, DT_STRING is implied for all Tensors. 

'repeated' is currently only supported when data_format = BigQueryClient.DataFormat.AVRO (which is default).

```python
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow_io.bigquery import BigQueryClient
from tensorflow_io.bigquery import BigQueryReadSession

GCP_PROJECT_ID = '<FILL_ME_IN>'
DATASET_GCP_PROJECT_ID = "bigquery-public-data"
DATASET_ID = "certain_dataset"
TABLE_ID = "certain_table_with_repeated_field"

def main():
  ops.enable_eager_execution()
  client = BigQueryClient()
  read_session = client.read_session(
      "projects/" + GCP_PROJECT_ID,
      DATASET_GCP_PROJECT_ID, TABLE_ID, DATASET_ID,
      selected_fiels={
          "field_a_name": {"mode": BigQueryClient.FieldMode.REPEATED, output_type: dtypes.int64},
          "field_b_name": {"mode": BigQueryClient.FieldMode.NULLABLE, output_type: dtypes.string},
          "field_c_name": {"mode": BigQueryClient.FieldMode.REQUIRED, output_type: dtypes.string}
          "field_d_name": {"mode": BigQueryClient.FieldMode.REPEATED, output_type: dtypes.string}
        }
      requested_streams=2,
      row_restriction="num_characters > 1000",
      data_format=BigQueryClient.DataFormat.AVRO)
  dataset = read_session.parallel_read_rows()

  row_index = 0
  for row in dataset.prefetch(10):
    print("row %d: %s" % (row_index, row))
    row_index += 1

if __name__ == '__main__':
  app.run(main)
```

Then each field of a repeated column becomes a rank-1 variable length Tensor. If you want to 
work that Tensor with dataset.batch, then you can use code like

```python
dataset = read_session.parallel_read_rows()
def sparse_dataset_map(features, sparse_column_names):
  """
  Map repeated columns to tf.SparseTensor.
  This matches how VarLenFeature is decoded from tf.Example datasets.
  """
  for col_name in sparse_column_names:
    l = tf.size(features[col_name], tf.int64)
    indices = tf.reshape(tf.range(l, dtype=tf.int64), [l, 1])
    features[col_name] = tf.SparseTensor(indices=indices,
                                         values=features[col_name],
                                         dense_shape=[l])
dataset_can_be_batched = dataset.map(lambda features: sparse_dataset_map(features, ["field_a_name", "field_d_name"]))
```
to map that that as a SparseTensor first, then dataset.batch can work. The behavior of returning this kind of SparseTensor is exactly aligned with how to decode tf.example tf.io.VarLenFeature,
which essentially just like the repeated column in BigQuery.




Please refer to BigQuery connector Python docstrings and to
[Enable BigQuery Storage API](https://cloud.google.com/bigquery/docs/reference/storage/rpc/)
documentation for more details about each parameter.
