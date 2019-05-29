# TensorFlow I/O Apache Arrow Datasets

Apache Arrow is a standard for in-memory columnar data, see [here](https://arrow.apache.org)
for more information on the project. An Arrow dataset makes it easy to bring in
column-oriented data from other systems to TensorFlow using the following
sources:

## From a Pandas DataFrame

An `ArrowDataset` can be made directly from an existing Pandas DataFrame, or
pyarrow record batches, in a Python process. Tensor types and shapes can be
inferred from the DataFrame, although currently only scalar and vector values
with primitive types are supported. PyArrow must be installed to use this
Dataset. Example usage:

```python
import tensorflow as tf
from tensorflow_io.arrow import ArrowDataset

# Assume `df` is an existing Pandas DataFrame
dataset = ArrowDataset.from_pandas(df)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
  for i in range(len(df)):
    print(sess.run(next_element))
```

NOTE: The entire DataFrame will be serialized to the Dataset and is not
recommended for use with large amounts of data

## From Arrow Feather Files

Feather is a light-weight file format that provides a simple and efficient way
to write Pandas DataFrames to disk, see [here](https://arrow.apache.org/docs/python/ipc.html#feather-format)
for more information and limitations of the format. An `ArrowFeatherDataset`
can be created to read one or more Feather files from the given pathnames. The
following example shows how to write a feather file from a Pandas DataFrame,
then read multiple files back as an `ArrowFeatherDataset`:

```python
from pyarrow.feather import write_feather

# Assume `df` is an existing Pandas DataFrame with dtypes=(int32, float32)
write_feather(df, '/path/to/a.feather')
```

```python
import tensorflow as tf
from tensorflow_io.arrow import ArrowFeatherDataset

# Each Feather file must have the same column types, here we use the above
# DataFrame which has 2 columns with dtypes=(int32, float32)
dataset = ArrowFeatherDataset(
    ['/path/to/a.feather', '/path/to/b.feather'],
    columns=(0, 1),
    output_types=(tf.int32, tf.float32),
    output_shapes=([], []))

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

# This will iterate over each row of each file provided
with tf.Session() as sess:
  while True:
    try:
      print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
      break
```

An alternate constructor can also be used to infer output types and shapes from
a given `pyarrow.Schema`, e.g. `dataset = ArrowFeatherDataset.from_schema(filenames, schema)`

## From a Stream of Arrow Record Batches

The `ArrowStreamDataset` provides a Dataset that will connect to a host over
a socket that is serving Arrow record batches in the Arrow stream format. See
[here](https://arrow.apache.org/docs/python/ipc.html#writing-and-reading-streams)
for more on the stream format. The following example will create an
`ArrowStreamDataset` that will connect to a host that is serving an Arrow
stream of record batches with 2 columns of dtypes=(int32, float32):

```python
import tensorflow as tf
from tensorflow_io.arrow import ArrowStreamDataset

# The str `host` should be in the format '<HOSTNAME>:<PORT>'
dataset = ArrowStreamDataset(
    host,
    columns=(0, 1),
    output_types=(tf.int32, tf.float32),
    output_shapes=([], []))

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

# The host connection is made when the Dataset op is run and will iterate over
# each row of each record batch until the Arrow stream is finished
with tf.Session() as sess:
  while True:
    try:
      print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
      break
```

An alternate constructor can also be used to infer output types and shapes from
a given `pyarrow.Schema`, e.g. `dataset = ArrowStreamDataset.from_schema(host, schema)`

## Creating Batches with Arrow Datasets

Arrow Datasets have optional parameters to specify a `batch_size` and
`batch_mode`. Supported `batch_modes` are: 'keep_remainder', 'drop_remainder'
and 'auto'. If the last elements of the Dataset do not combine to the set
`batch_size`, then 'keep_remainder' will return a partial batch, while
'drop_remainder' will discard the partial batch. Setting `batch_mode` to 'auto'
will automatically set a batch size to the number of records in the incoming
Arrow record batches. This a good option to use if the incoming Arrow record
batch size can be controlled to ensure the output batch size is not too large
and sequential Arrow record batches are sized equally.

Setting the `batch_size` or using `batch_mode` of 'auto' can be more efficient
than using `tf.data.Dataset.batch()` on an Arrow Dataset. This is because the
output tensor can be sized to the desired batch size on creation, and then data
is transferred directly from Arrow memory. Otherwise, if batching elements with
the output of an Arrow Dataset, e.g. `ArrowDataset(...).batch(batch_size=4)`,
then the tensor data will need to be aggregated and copied to get the final
batched outputs.
