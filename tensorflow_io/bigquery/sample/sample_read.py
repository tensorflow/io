""" Sample for reading data from BigQuery.

Run it as: blaze run [-c dbg] -- --gcp_project_id=<GCP_PROJECT> [--alsologtostderr] [--vmodule=bigquery*=3]
"""


from absl import app
from absl import flags
print("started BigQuery read sample")
from tensorflow.python.framework import ops
print("imported tensorflow")
from tensorflow.python.framework import dtypes
from tensorflow_io.bigquery.python.ops.bigquery_api import BigQueryClient
from tensorflow_io.bigquery.python.ops.bigquery_api import BigQueryReadSession
print("imported BigQuery")

FLAGS = flags.FLAGS

flags.DEFINE_string("gcp_project_id",
                    "alekseyv-scalableai-test",
                    "GCP project id")

GCP_PROJECT_ID = "bigquery-public-data"
DATASET_ID = "samples"
TABLE_ID = "wikipedia"

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  ops.enable_eager_execution()
  print("initializing BigQueryClient")
  client = BigQueryClient()
  parent = "projects/" + FLAGS.gcp_project_id
  read_session = client.read_session(
      parent,
      GCP_PROJECT_ID, TABLE_ID, DATASET_ID,
      ["title", "id", "num_characters"],
      [dtypes.string, dtypes.int64, dtypes.int64],
      requested_streams=2)
  dataset = read_session.parallel_read_rows()

  print("retrieving rows:")
  row_index = 0
  for row in dataset.skip(10).take(10).repeat(3):
    print(">>>>>> row %d: %s" % (row_index, row))
    row_index += 1

  print("finished reading data")


if __name__ == '__main__':
  app.run(main)
