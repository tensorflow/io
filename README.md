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
- `tensorflow_io.mnist`: MNIST file format support.
- `tensorflow_io.pubsub`: Google Cloud Pub/Sub support.
- `tensorflow_io.oss`: Alibaba Cloud Object Storage Service (OSS) support. Usage guide [here](https://github.com/tensorflow/io/blob/master/tensorflow_io/oss/README.md).

## Installation

### Python Package

The `tensorflow-io` Python package could be installed with pip directly:
```sh
$ pip install tensorflow-io
```

People who are a little more adventurous can also try our nightly binaries:
```sh
$ pip install tensorflow-io-nightly
```

The use of tensorflow-io is straightforward with keras. Below is the example
of [Get Started with TensorFlow](https://www.tensorflow.org/tutorials) with
data processing replaced by tensorflow-io:

```python
import tensorflow as tf
import tensorflow_io.mnist as mnist_io

# Read MNIST into tf.data.Dataset
d_train = mnist_io.MNISTDataset(
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz')

# By default image data is uint8 so conver to float32.
d_train = d_train.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y)).batch(1)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(d_train, epochs=5, steps_per_epoch=10000)
```

Note that in the above example, [MNIST](http://yann.lecun.com/exdb/mnist/) database
files are assumed to have been downloaded and saved to the local directory.
Compression files (e.g. gzip) could be detected and uncompressed automatically.

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

### TensorFlow Version Compatibility

To ensure compatibility with TensorFlow, it is recommended to install a matching
version of TensorFlow I/O according to the table below:

| TensorFlow I/O Version | TensorFlow Compatibility | Release Date |
| --- | --- | --- |
| 0.5.0 | 1.13.x | Apr 12, 2019 |
| 0.4.0 | 1.13.x | Mar 01, 2019 |
| 0.3.0 | 1.12.0 | Feb 15, 2019 |
| 0.2.0 | 1.12.0 | Jan 29, 2019 |
| 0.1.0 | 1.12.0 | Dec 16, 2018 |

## Developing

### Python

For Python development, a reference Dockerfile [here](dev/Dockerfile) can be
used to build the TensorFlow I/O package (`tensorflow-io`) from source:
```sh
$ # Build and run the Docker image
$ docker build -f dev/Dockerfile -t tfio-dev .
$ docker run -it --rm --net=host -v ${PWD}:/v -w /v tfio-dev
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

```sh
$ bash -x -e tests/test_ignite/start_ignite.sh
$ bash -x -e tests/test_kafka/kafka_test.sh start kafka
$ bash -x -e tests/test_kinesis/kinesis_test.sh start kinesis
```

#### Running Python Style Checks

Style checks for Python can be run with the following commands:

```sh
$ curl -o .pylint -sSL https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc
$ find . -name '*.py' | xargs pylint --rcfile=.pylint
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

## Community

* SIG IO [Google Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/io) and mailing list: [io@tensorflow.org](io@tensorflow.org)
* SIG IO [Monthly Meeting Notes](https://docs.google.com/document/d/1CB51yJxns5WA4Ylv89D-a5qReiGTC0GYum6DU-9nKGo/edit)
* Gitter room: [tensorflow/sig-io](https://gitter.im/tensorflow/sig-io)

## License

[Apache License 2.0](LICENSE)
