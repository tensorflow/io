<div align="center">
  <img src="https://github.com/tensorflow/community/blob/master/sigs/logos/SIGIO.png" width="60%"><br><br>
</div>

-----------------

# TensorFlow I/O

[![GitHub CI Status Badge](https://github.com/tensorflow/io/workflows/GitHub%20CI/badge.svg?branch=master)](https://github.com/tensorflow/io/actions?query=branch%3Amaster)
[![PyPI Status Badge](https://badge.fury.io/py/tensorflow-io.svg)](https://pypi.org/project/tensorflow-io/)
[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/tfio)](https://cran.r-project.org/package=tfio)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/io)

TensorFlow I/O is a collection of file systems and file formats that are not
available in TensorFlow's built-in support. A full list of supported file systems
and file formats by TensorFlow I/O can be found [here](https://www.tensorflow.org/io/api_docs/python/tfio).

The use of tensorflow-io is straightforward with keras. Below is the example
of [Get Started with TensorFlow](https://www.tensorflow.org/tutorials) with
data processing replaced by tensorflow-io:

```python
import tensorflow as tf
import tensorflow_io as tfio

# Read MNIST into Dataset
d_train = tfio.IODataset.from_mnist(
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz').batch(1)

# By default image data is uint8 so conver to float32.
d_train = d_train.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y))

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
files' URL address are directly passes to `tfio.IODataset.from_mnist`, the API
used to create MNIST Dataset. We are able to do that because `tensorflow-io`
support `HTTP` file system out of the box. There is no need to download and
save files to local directory any more. Note we are also passing the compressed
files (gzip) as is, since `tensorflow-io` is able to detect and uncompress
automatically for MNIST dataset if needed.

Please check the official [documentation](https://www.tensorflow.org/io) for more
detailed usages.

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
| 0.11.0 | 2.1.x | Jan 10, 2019 |
| 0.10.0 | 2.0.x | Dec 5, 2019 |
| 0.9.1 | 2.0.x | Nov 15, 2019 |
| 0.9.0 | 2.0.x | Oct 18, 2019 |
| 0.8.1 | 1.15.x | Nov 15, 2019 |
| 0.8.0 | 1.15.x | Oct 17, 2019 |
| 0.7.2 | 1.14.x | Nov 15, 2019 |
| 0.7.1 | 1.14.x | Oct 18, 2019 |
| 0.7.0 | 1.14.x | Jul 14, 2019 |
| 0.6.0 | 1.13.x | May 29, 2019 |
| 0.5.0 | 1.13.x | Apr 12, 2019 |
| 0.4.0 | 1.13.x | Mar 01, 2019 |
| 0.3.0 | 1.12.0 | Feb 15, 2019 |
| 0.2.0 | 1.12.0 | Jan 29, 2019 |
| 0.1.0 | 1.12.0 | Dec 16, 2018 |

### Build Status and CI

| Build | Status |
| --- | --- |
| Linux CPU Python 2 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-py2.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-py2.html) |
| Linux CPU Python 3 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-py3.html) |
| Linux GPU Python 2| [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-gpu-py2.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-gpu-py2.html) |
| Linux GPU Python 3| [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-gpu-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-gpu-py3.html) |

Because of manylinux2010 requirement, TensorFlow I/O is built with
Ubuntu:16.04 + Developer Toolset 7 (GCC 7.3) on Linux. Configuration
with Ubuntu 16.04 with Developer Toolset 7 is not exactly straightforward.
If the system have docker installed, then the following command
will automatically build manylinux2010 compatible whl package:

```sh
bash -x -e .travis/python.release.sh
```

It takes some time to build, but once complete, there will be python
`2.7`, `3.5`, `3.6`, `3.7` compatible whl packages available in `wheelhouse`
directory.

On macOS, the same command could be used though the script expect `python` in shell
and will only generate a whl package that matches the version of `python` in shell. If
you want to build a whl package for a specific python then you have to alias this version
of python to `python` in shell.

Note the above command is also the command we use when releasing packages for Linux and macOS.

TensorFlow I/O uses both Travis CI and Google CI (Kokoro) for continuous integration.
Travis CI is used for macOS build and test. Kokoro is used for Linux build and test.
Again, because of the manylinux2010 requirement, on Linux whl packages are always
built with Ubuntu 16.04 + Developer Toolset 7. Tests are done on a variatiy of systems
with different python version to ensure a good coverage:

| Python | Ubuntu 16.04| Ubuntu 18.04 | macOS + osx9 |
| ------- | ----- | ------- | ------- |
| 2.7 |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| 3.5 |  :heavy_check_mark: | N/A | :heavy_check_mark: |
| 3.6 |  N/A | :heavy_check_mark: | :heavy_check_mark: |
| 3.7 |  N/A | :heavy_check_mark: | N/A |


TensorFlow I/O has integrations with may systems and cloud vendors such as
Prometheus, Apache Kafka, Apache Ignite, Google Cloud PubSub, AWS Kinesis,
Microsoft Azure Storage, Alibaba Cloud OSS etc.

We tried our best to test against those systems in our continuous integration
whenever possible. Some tests such as Prometheus, Kafka, and Ignite
are done with live systems, meaning we install Prometheus/Kafka/Inite on CI machine before
the test is run. Some tests such as Kinesis, PubSub, and Azure Storage are done
through official or non-official emulators. Offline tests are also performed whenever
possible, though systems covered through offine tests may not have the same
level of coverage as live systems or emulators.


|  | Live System | Emulator| CI Integration |  Offline |
| ------- | ----- | ----- | ----- | ----- |
| Apache Kafka | :heavy_check_mark:  | | :heavy_check_mark:| |
| Apache Ignite |  :heavy_check_mark: | |:heavy_check_mark:| |
| Prometheus |  :heavy_check_mark: | |:heavy_check_mark:| |
| Google PubSub |   | :heavy_check_mark: |:heavy_check_mark:| |
| Azure Storage |   | :heavy_check_mark: |:heavy_check_mark:| |
| AWS Kinesis |   | :heavy_check_mark: |:heavy_check_mark:| |
| Alibaba Cloud OSS |   | | |  :heavy_check_mark: |
| Google BigTable/BigQuery |   | to be added | | |

Note:
- Offical [PubSub Emulator](https://cloud.google.com/sdk/gcloud/reference/beta/emulators/pubsub/) by Google Cloud for Cloud PubSub.
- Official [Azurite Emulator](https://github.com/Azure/Azurite) by Azure for Azure Storage.
- None-official [LocalStack emulator](https://github.com/localstack/localstack) by LocalStack for AWS Kinesis.


## Development

### Python

For Python development, a reference Dockerfile [here](tools/dev/Dockerfile) can be
used to build the TensorFlow I/O package (`tensorflow-io`) from source:
```sh
$ # Build and run the Docker image
$ docker build -f tools/dev/Dockerfile -t tfio-dev .
$ docker run -it --rm --net=host -v ${PWD}:/v -w /v tfio-dev
$ # In Docker, configure will install TensorFlow or use existing install
$ ./configure.sh
$ # Build TensorFlow I/O C++. For compilation optimization flags, the default (-march=native) optimizes the generated code for your machine's CPU type. [see here](https://www.tensorflow.org/install/source#configuration_options)
$ bazel build -c opt --copt=-march=native --copt=-fPIC -s --verbose_failures //tensorflow_io/...
$ # Run tests with PyTest, note: some tests require launching additional containers to run (see below)
$ pytest -s -v tests/
$ # Build the TensorFlow I/O package
$ python setup.py bdist_wheel
```

A package file `dist/tensorflow_io-*.whl` will be generated after a build is successful.

NOTE: When working in the Python development container, an environment variable
`TFIO_DATAPATH` is automatically set to point tensorflow-io to the shared C++
libraries built by Bazel to run `pytest` and build the `bdist_wheel`. Python
`setup.py` can also accept `--data [path]` as an argument, for example
`python setup.py --data bazel-bin bdist_wheel`.

NOTE: While the tfio-dev container gives developers an easy to work with
environment, the released whl packages are build differently due to manylinux2010
requirements. Please check [Build Status and CI] section for more details
on how the released whl packages are generated.

#### Starting Test Containers

Some tests require launching a test container before running. In order
to run all tests, execute the following commands:

```sh
$ bash -x -e tests/test_ignite/start_ignite.sh
$ bash -x -e tests/test_kafka/kafka_test.sh start kafka
$ bash -x -e tests/test_kinesis/kinesis_test.sh start kinesis
```

#### Running Python and Bazel Style Checks

Style checks for Python and Bazel can be run with the following commands
(docker has to be available):

```sh
$ bash -x -e .travis/lint.sh
```

In case there are any Bazel style errors, the following command could be invoked 
to fix and Bazel style issues:

```sh
$ docker run -i -t --rm -v $PWD:/v -w /v --net=host golang:1.12 bash -x -e -c 'go get github.com/bazelbuild/buildtools/buildifier && buildifier $(find . -type f \( -name WORKSPACE -or -name BUILD -or -name *.BUILD \))'
```

After the command is run, any Bazel files with style issues will have been modified and corrected.

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

## Contributing

Tensorflow I/O is a community led open source project. As such, the project
depends on public contributions, bug-fixes, and documentation. Please
see [contribution guidelines](CONTRIBUTING.md) for a guide on how to
contribute.

## Community

* SIG IO [Google Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/io) and mailing list: [io@tensorflow.org](io@tensorflow.org)
* SIG IO [Monthly Meeting Notes](https://docs.google.com/document/d/1CB51yJxns5WA4Ylv89D-a5qReiGTC0GYum6DU-9nKGo/edit)
* Gitter room: [tensorflow/sig-io](https://gitter.im/tensorflow/sig-io)

## More Information

* [TensorFlow with Apache Arrow Datasets](https://medium.com/tensorflow/tensorflow-with-apache-arrow-datasets-cdbcfe80a59f) - [Bryan Cutler](https://github.com/BryanCutler)
* [How to build a custom Dataset for Tensorflow](https://towardsdatascience.com/how-to-build-a-custom-dataset-for-tensorflow-1fe3967544d8) - [Ivelin Ivanov](https://github.com/ivelin)
* [TensorFlow on Apache Ignite](https://medium.com/tensorflow/tensorflow-on-apache-ignite-99f1fc60efeb) - [Anton Dmitriev](https://github.com/dmitrievanthony)

## License

[Apache License 2.0](LICENSE)
