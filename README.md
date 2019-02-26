# TensorFlow I/O

[![Travis-CI Build Status](https://travis-ci.org/tensorflow/io.svg?branch=master)](https://travis-ci.org/tensorflow/io)
[![PyPI Status Badge](https://badge.fury.io/py/tensorflow-io.svg)](https://pypi.org/project/tensorflow-io/)
[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/tfio)](https://cran.r-project.org/package=tfio)

TensorFlow I/O is a collection of file systems and file formats that are not
available in TensorFlow's built-in support.

At the moment TensorFlow I/O supports the following data sources:
- `tensorflow_io.ignite`: Data source for Apache Ignite and Ignite File System (IGFS). Overview and usage guide [here](tensorflow_io/ignite/README.md).
- `tensorflow_io.kafka`: Apache Kafka stream-processing support.
- `tensorflow_io.kinesis`: Amazon Kinesis data streams support.
- `tensorflow_io.hadoop`: Hadoop SequenceFile format support.
- `tensorflow_io.arrow`: Apache Arrow data format support. Usage guide [here](tensorflow_io/arrow/README.md).
- `tensorflow_io.image`: WebP and TIFF image format support.
- `tensorflow_io.libsvm`: LIBSVM file format support.
- `tensorflow_io.video`: Video file support with FFmpeg.
- `tensorflow_io.parquet`: Apache Parquet data format support.
- `tensorflow_io.lmdb`: LMDB file format support.

## Installation

### Python Package

The `tensorflow-io` Python package could be installed with pip directly:
```
$ pip install tensorflow-io
```

People who are a little more adventurous can also try our nightly binaries:
```
$ pip install tensorflow-io-nightly
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

### R Package

Once the `tensorflow-io` Python package has beem successfully installed, you
can then install the latest stable release of the R package via:

```r
install.packages('tfio')
```

You can also install the development version from Github via:
```r
if (!require("devtools")) install.packages("devtools")
devtools::install_github("tensorflow/io", subdir = "R-package")
```

## Developing

### Python

For Python development, a reference Dockerfile [here](dev/Dockerfile) can be
used to build the TensorFlow I/O package (`tensorflow-io`) from source:
```sh
$ # Build and run the Docker image
$ docker build -f dev/Dockerfile -t tfio-dev .
$ docker run -it -rm --net=host -v ${PWD}:/v -w /v tfio-dev
$ # In Docker, configure will install TensorFlow or use existing install
$ ./configure.sh
$ # Build TensorFlow I/O C++
$ bazel build -s --verbose_failures //tensorflow_io/...
$ # Run tests with PyTest, note: some tests require launching additional containers to run (see below)
$ pytest tests/
$ # Build the TensorFlow I/O package
$ python setup.py bdist_wheel
```

A package file `dist/tensorflow_io-*.whl` will be generated after a build is successful.

NOTE: When working in the Python development container, an environment variable
`TFIO_DATAPATH` is automatically set to point tensorflow-io to the shared C++
libraries built by Bazel to run `pytest` and build the `bdist_wheel`. Python
`setup.py` can also accept `--data [path]` as an argument, for example
`python setup.py --data bazel-bin bdist_wheel`.

#### Starting Test Containers

Some tests require launching a test container before running. In order
to run all tests, execute the following commands:

```
$ bash -x -e tests/test_ignite/start_ignite.sh
$ bash -x -e tests/test_kafka/kafka_test.sh start kafka
$ bash -x -e tests/test_kinesis/kinesis_test.sh start kinesis
```

### R

We provide a reference Dockerfile [here](R-package/scripts/Dockerfile) for you
so that you can use the R package directly for testing. You can build it via:
```sh
docker build -t tfio-r-dev -f R-package/scripts/Dockerfile .
```

Inside the container, you can start your R session, instantiate a `SequenceFileDataset`
from an example [Hadoop SequenceFile](https://wiki.apache.org/hadoop/SequenceFile)
[string.seq](R-package/tests/testthat/testdata/string.seq), and then use any [transformation functions](https://tensorflow.rstudio.com/tools/tfdatasets/articles/introduction.html#transformations) provided by [tfdatasets package](https://tensorflow.rstudio.com/tools/tfdatasets/) on the dataset like the following:

```r
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
