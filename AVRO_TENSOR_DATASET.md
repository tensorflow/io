# AvroTensorDataset

AvroTensorDataset is a `tf.data.Dataset` implementation. It consumes records from one or more Avro files. The supported schema are discussed with more details in later section.

AvroTensorDataset loads Avro records from files into a dict of tensors.
The output dict has feature name as key and tf.Tensor or tf.SparseTensor
as value. The output tensor values are batched with the user defined
batch size. 

## Python API

A minimal example is given below:

    >>> import tempfile
    >>> import avro.schema
    >>> from avro.datafile import DataFileWriter
    >>> from avro.io import DatumWriter
    >>> from tensorflow_io.python.experimental.atds.dataset import ATDSDataset
    >>> from tensorflow_io.python.experimental.atds.features import DenseFeature
    >>> example_path = os.path.join(tempfile.gettempdir(), "example.avro")
    >>> np.random.seed(0)

    >>> # Define Avro schema 
    >>> json_schema = '''{
    ...     "type": "record",
    ...     "name": "example",
    ...     "fields": [
    ...         { "name": "x", "type": "float" },
    ...         { "name": "y", "type": "float" }
    ...     ]
    ... }'''
    >>> schema = avro.schema.Parse(json_schema)

    >>> # Write the Avro records to a file.
    >>> with open(example_path, "wb") as f:
    ...     writer = DataFileWriter(f, DatumWriter(), schema)
    ...     for _ in range(3):
    ...         x, y = np.random.random(), np.random.random()
    ...         writer.append({"x": x, "y": y})
    ...     writer.close()

    >>> # Read the data back out.
    >>> feature_config = {
    ...     "x": DenseFeature([], dtype=tf.float32),
    ...     "y": DenseFeature([], dtype=tf.float32)
    ... }
    >>> for batch in ATDSDataset([example_path], batch_size=2,
    ...                         features=feature_config):
    ...     print("x = {x},  y = {y}".format(**batch))
    x = [0.5488135  0.60276335],  y = [0.71518934 0.5448832 ]
    x = [0.4236548],  y = [0.6458941]

The constructor supports the following arguments:
| Argument | type | comment |
|---|---|---|
| filenames | tf.string or tf.data.Dataset | A tf.string tensor containing one or more filenames. |
| batch_size | tf.int64 | A tf.int64 scalar representing the number of records to read and parse per iteration. |
| features | Dict[str, Union[<br>  DenseFeature,     <br>  SparseFeature, <br>  VarlenFeature]] | A feature configuration dict with feature name as key and feature spec as value. We support DenseFeature, SparseFeature, and VarlenFeature specs. All of them are named tuples with shape and dtype information. |
| drop_remainder | tf.bool | (Optional.) A tf.bool scalar tf.Tensor, representing whether the last batch should be dropped in the case it has fewer than batch_size elements. The default behavior is not to drop the smaller batch. |
| reader_buffer_size | tf.int64 | (Optional) A tf.int64 scalar representing the number of bytes used in the file content buffering. Default is 128 * 1024 (128KB). |
| shuffle_buffer_size | tf.int64 | (Optional) A tf.int64 scalar representing the number of records to shuffle together before batching. Default is zero. Zero shuffle <br>buffer size means shuffle is disabled. |
| num_parallel_calls | tf.int64 | (Optional) A tf.int64 scalar representing the maximum thread number used in the dataset. If greater than one, records in files are processed in parallel. The number will be truncated when it is greater than the maximum available parallelism number on the host. If the value tf.data.AUTOTUNE is used, then the number of parallel calls is set dynamically based on available CPU and workload. Default is 1. |

At a minimum, the constructor requires the list of files to read, the batch size (to support batching), and dict containing feature specs. Prefetch is enabled by default and whose behavior can be tuned via reader_buffer_size. Parsing happens automatically within the ATDSDataset operation. Shuffling is supported via configuring shuffle_buffer_size.

## Supported Avro Schemas

Although Avro supports many complex types (unions, maps, etc.), AvroTensorDataset only supports records of primitives and nested arrays. These supported types cover most TensorFlow use cases, and we get a big performance boost by only supporting a subset of complex types (more on that later).

AvroTensorDataset supports dense features, sparse features, and varlen features. It also supports certain TensorFlow primitives that are supported by Avro. They are represented in Avro via the following:

### Primitive Types

All Avro primitive types are supported, and map to the following TensorFlow dtypes:

    +-------------------+----------------+
    |  Avro data type   |  TF data type  |
    |-------------------|----------------|
    |        int        |    tf.int32    |
    |-------------------|----------------|
    |        long       |    tf.int64    |
    |-------------------|----------------|
    |        float      |   tf.float32   |
    |-------------------|----------------|
    |        double     |   tf.float64   |
    |-------------------|----------------|
    |        string     |   tf.string    |
    |-------------------|----------------|
    |        bool       |    tf.bool     |
    |-------------------|----------------|
    |        bytes      |    tf.string   |  
    |-------------------|----------------|

### Dense Features

Dense features are represented as nested arrays in Avro. For example, a doubly nested array represents a dense feature with rank 2. Some examples of Avro schemas representing dense features:
```json
"fields": [
  { 
    "name" : "scalar_double_feature", 
    "type" : "double"
  },
  {
    "name" : "1d_double_feature",
    "type" : {
      "type": "array",
      "items" : "double"
    }
  },
  {
    "name" : "2d_float_feature",
    "type" : {
      "type": "array",
      "items" : {
        "type": "array",
        "items": "float"
      }
    }
  },
  {
    "name" : "3d_int_feature",
    "type" : {
      "type": "array",
      "items" : {
        "type": "array",
        "items":  {
          "type": "array",
          "items": "int"
        }
      }
    }
  }
]
```
Dense features are parsed into dense tensors. For the above, the features argument to ATDSDataset might be:
```python
{
    "scalar_double_feature": DenseFeature(shape=[], dtype=tf.float64),
    "1d_double_feature": DenseFeature(shape=[128], dtype=tf.float64),
    "2d_float_feature": DenseFeature(shape=[16, 100], dtype=tf.float32),
    "3d_int_feature": DenseFeature(shape=[8, 10, 20], dtype=tf.int32),
}
```

### Sparse Features
Sparse features are represented as a flat list of arrays in Avro. For a sparse feature with rank N, the Avro schema contains N+1 arrays: arrays named “indices0”, “indices1”, …, “indices(N-1)” and an array named “values”. All N+1 arrays should have the same length. For example, this is the schema for a sparse feature with dtype float, and rank 2:
```json
"fields": [
  {
    "name" : "2d_float_sparse_feature",
    "type" : {
      "type" : "record",
      "name" : "2d_float_sparse_feature",
      "fields" : [ {
          "name": "indices0",
          "type": {
            "type": "array",
            "items": "long"
          }
        }, {
          "name": "indices1",
          "type": {
            "type": "array",
            "items": "long"
          }
        }, {
          "name": "values",
          "type": {
            "type": "array",
            "items": "float"
          }
        }
      ]
    }
  }
]
```

Sparse features are parsed into sparse tensors. For the above, the features argument to ATDSDataset might be:
```python
{
    "2d_float_sparse_feature": SparseFeature(shape=[16, 10], dtype=tf.float32),
}
```
The i-th indices array represents the indices for rank i, i.e. the Avro representation for a sparse tensor is in coordinate format. For example, the sparse tensor: tf.sparse.SparseTensor(indices=[[0,1], [2,4], [6,5]], values=[1.0, 2.0, 3.0], dense_shape=[8, 10]) would be represented in Avro via the following:
```json
{
  "indices0" : [0, 2, 6],
  "indices1" : [1, 4, 5],
  "values" : [1.0, 2.0, 3.0]
}
```

### Varlen Features
Varlen features are similar to dense features in that they are also represented as nested arrays in Avro, but they can have dimensions of unknown length (indicated by -1). Some examples of Avro schemas representing varlen features:
```json
"fields": [
  {
    "name" : "1d_bool_varlen_feature",
    "type" : {
      "type": "array",
      "items" : "boolean"
    }
  },
  {
    "name" : "2d_long_varlen_feature",
    "type" : {
      "type": "array",
      "items" : {
        "type": "array",
        "items": "long"
      }
    }
  },
  {
    "name" : "3d_int_varlen_feature",
    "type" : {
      "type": "array",
      "items" : {
        "type": "array",
        "items":  {
          "type": "array",
          "items": "int"
        }
      }
    }
  }
]
```
Dimensions with length -1 can be variable length, hence varlen features are parsed into sparse tensors. For the above, the features argument to ATDSDataset might be:
```python
{
    "1d_bool_varlen_feature": VarlenFeature(shape=[-1], dtype=tf.bool),
    "2d_long_varlen_feature": VarlenFeature(shape=[2, -1], dtype=tf.int64),
    "3d_int_varlen_feature": VarlenFeature(shape=[-1, 10, -1], dtype=tf.int32),
}
```
Here, 2d_long_varlen_feature has variable length in rank 1; for example, an object with values [[1, 2, 3], [4, 5]] would be parsed as tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]], values=[1, 2, 3, 4, 5], dense_shape=[2, 3]).

## Input files
### Single or multiple input files
AvroTensorDataset can process single or multiple files together. If there's no scheme e.g. `hdfs://` in the file path,
it will search for local file. For example,

    # Single local input file
    input_file = "path/to/file"
    dataset = ATDSDataset(input_file, batch_size=2, features=feature_config)

    # Single input file on HDFS
    input_file = "hdfs://default/path/to/file"
    dataset = ATDSDataset(input_file, batch_size=2, features=feature_config)

    # Multiple input files
    input_files = ["path/to/file1", "path/to/file2", ...]
    dataset = ATDSDataset(input_files, batch_size=2, features=feature_config)

### Inputs with file pattern
To read input with specific file pattern, users can use `tf.data.Dataset.list_files` to get the file glob,
and process them with AvroTensorDataset. For example,

    filenames = tf.data.Dataset.list_files(file_pattern="path/to/files/*.avro")

    # Batch all file names so that we can process them together.
    file_num = filenames.cardinality()
    dataset = filenames.batch(file_num)

    dataset = dataset.interleave(
        lambda filename: ATDSDataset(filenames=filename, ...),
        cycle_length=1
    )

Moreover, users can batch files with fixed number and leverage `tf.data.Dataset.interleave` to process them in
parallel. For example,

    filenames = tf.data.Dataset.list_files(file_pattern="path/to/files/*.avro")

    # Batch 10 files together so that we can process 10 files together.
    dataset = filenames.batch(10)

    # Launch 4 interleave threads and each thread will process 10 files with AvroTensorDataset in parallel.
    dataset = dataset.interleave(
        lambda filename: ATDSDataset(filenames=filename, ...),
        cycle_length=4,
        num_parallel_calls=4
    )

## Batch

The output tensor values are always batched with the user defined batch size. If the last batch does not have
enough data to batch, whatever remains will be batched with smaller batch size. User can drop the last small
batch by setting drop_remainder to true.

    # Drop the last small batch so that every batch has the same batch size.
    dataset = ATDSDataset(filenames, batch_size=128, features=feature_config, drop_remainder=True)

## Shuffle

Shuffle can be enabled before batching by configuring shuffle buffer size. The shuffle buffer size dictates the
elements *in addition* to the batch size that would be read and sampled. For example,

    # Shuffle records in a buffer with size 1024 before batching.
    dataset = ATDSDataset(filenames, batch_size=128, features=feature_config, shuffle_buffer_size=1024)

Shuffle is disabled by default with shuffle_buffer_size equals 0.

Internally, AvroTensorDataset keeps collecting Avro blocks(a sequence of Avro records), until the total number of unread
records is greater than the shuffle buffer size + batch_size, then randomly samples block from the collected blocks.
An Avro Record from the sampled block will be parsed and batched into the output tensors.

For instance, assume your dataset contains 5 blocks with 100 records in each block. When the batch size is set to
32 and shuffle buffer size is set to 128, this dataset will collect two blocks as the two blocks contains more
than 128 + 32 = 160 unread records, and randomly samples block from the two blocks 32 times. When a block is sampled,
a record in the sampled block is read and batched into the output tensor dict until all records in the sampled block
are read. If only one block fits into the batch + shuffle_buffer_size, records in that block will be read sequentially
without shuffle. Users can increase the shuffle buffer size or apply dataset unbatch, shuffle, and batch for better
shuffling.

## Parallel computing

Batching, shuffling, and record parsing can be done in parallel by configuring the num_parallel_calls in AvroTensorDataset.
num_parallel_calls controls the number of threads for processing the input files. For example, if users want to
do batching, shuffling, and parsing in parallel with four threads, they can configure AvroTensorDataset like this

    dataset = ATDSDataset(
        filenames,
        batch_size=128,
        features=feature_config,
        shuffle_buffer_size=1024,
        num_parallel_calls=4  # Processing data in parallel with 4 threads within ops.
    )

It is different from what we have seen in `interleave`. What num_parallel_calls controls is the number of threads
used in one AvroTensorDataset. Hence, if one uses 4 interleave node and each interleave node runs with
2 internal threads in AvroTensorDataset, the total number of launched threads will be 4 * 2 = 8. For example,

    filenames = tf.data.Dataset.list_files(file_pattern="path/to/files/*.avro")

    # Batch 10 files together so that we can process 10 files together.
    dataset = filenames.batch(10)

    # Launch 4 interleave threads and each thread will process 10 files with AvroTensorDataset in parallel.
    dataset = dataset.interleave(
        lambda filename: ATDSDataset(
            filenames=filename,
            batch_size=128,
            features=feature_config,
            shuffle_buffer_size=1024,
            num_parallel_calls=2),  # 2 threads for each interleave node.
        cycle_length=4,
        num_parallel_calls=4
    )

By default, AvroTensorDataset will use all available CPU cores on the host as its num_parallel_calls number.

