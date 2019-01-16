# TensorFlow I/O

[![Travis-CI Build Status](https://travis-ci.org/tensorflow/io.svg?branch=master)](https://travis-ci.org/tensorflow/io) 

TensorFlow I/O is a collection of file systems and file formats that are not
available in TensorFlow's built-in support.

At the moment TensorFlow I/O supports 5 data sources:
- `tensorflow_io.ignite`: Data source for Apache Ignite and Ignite File System (IGFS).
- `tensorflow_io.kafka`: Apache Kafka stream-processing support.
- `tensorflow_io.kinesis`: Amazon Kinesis data streams support.
- `tensorflow_io.hadoop`: Hadoop SequenceFile format support.
- `tensorflow_io.arrow`: Apache Arrow data format support.

## Installation

The `tensorflow-io` package could be installed with pip directly:
```
$ pip install tensorflow-io
```

The related module such as Kafka could be imported with python:
```
$  python
Python 2.7.6 (default, Nov 13 2018, 12:45:42)
[GCC 4.8.4] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> import tensorflow_io.kafka as kafka
>>>
>>> dataset = kafka.KafkaDataset(["test:0:0:4"], group="test", eof=True)
>>> iterator = dataset.make_initializable_iterator()
>>> init_op = iterator.initializer
>>> get_next = iterator.get_next()
>>>
>>> with tf.Session() as sess:
...   print(sess.run(init_op))
...   for i in range(5):
...     print(sess.run(get_next))
>>>
```

Note that python has to run outside of repo directory itself, otherwise python may not
be able to find the correct path to the module.

## Using TensorFlow I/O

### Apache Arrow Datasets

Apache Arrow is a standard for in-memory columnar data, see [here](https://arrow.apache.org)
for more information on the project. An Arrow dataset makes it easy to bring in
column-oriented data from other systems to TensorFlow using the following
sources:

#### Pandas DataFrame

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

#### Arrow Feather Dataset

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

### Arrow Stream Dataset

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

## Developing

### Python

The TensorFlow I/O package (`tensorflow-io`) could be built from source:
```sh
$ docker run -it -v ${PWD}:/working_dir -w /working_dir tensorflow/tensorflow:custom-op
$ # In docker
$ curl -OL https://github.com/bazelbuild/bazel/releases/download/0.20.0/bazel-0.20.0-installer-linux-x86_64.sh
$ chmod +x bazel-0.20.0-installer-linux-x86_64.sh
$ ./bazel-0.20.0-installer-linux-x86_64.sh
$ ./configure.sh
$ bazel build build_pip_pkg
$ bazel-bin/build_pip_pkg artifacts
```

A package file `artifacts/tensorflow_io-*.whl` will be generated after a build is successful.

### R

We provide a reference Dockerfile [here](R-package/scripts/Dockerfile) for you
so that you can use the R package directly for testing. You can build it via:
```
docker build -t tfio-r-dev -f R-package/scripts/Dockerfile .
```

Inside the container, you can start your R session, instantiate a `SequenceFileDataset`
from an example [Hadoop SequenceFile](https://wiki.apache.org/hadoop/SequenceFile)
[string.seq](R-package/tests/testthat/testdata/string.seq), and then use any [transformation functions](https://tensorflow.rstudio.com/tools/tfdatasets/articles/introduction.html#transformations) provided by [tfdatasets package](https://tensorflow.rstudio.com/tools/tfdatasets/) on the dataset like the following:

```{R}
library(tfio)
dataset <- sequence_file_dataset("R-package/tests/testthat/testdata/string.seq") %>%
    dataset_repeat(2)

sess <- tf$Session()
iterator <- make_iterator_one_shot(dataset)
next_batch <- iterator_get_next(iterator)

until_out_of_range({
  batch <- sess$run(next_batch)
  print(batch)
})
```

## License

[Apache License 2.0](LICENSE)
