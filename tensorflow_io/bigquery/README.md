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
      row_restriction="num_characters > 1000")
  dataset = read_session.parallel_read_rows()

  row_index = 0
  for row in dataset.prefetch(10):
    print("row %d: %s" % (row_index, row))
    row_index += 1

if __name__ == '__main__':
  app.run(main)

```

Please refer to BigQuery connector Python docstrings and to
[Enable BigQuery Storage API](https://cloud.google.com/bigquery/docs/reference/storage/rpc/)
documentation for more details about each parameter.
