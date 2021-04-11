<div align="center">
  <img src="https://github.com/tensorflow/community/blob/master/sigs/logos/SIGIO.png" width="60%"><br><br>
</div>

-----------------

# TensorFlow I/O

[![GitHub CI](https://github.com/tensorflow/io/workflows/GitHub%20CI/badge.svg?branch=master)](https://github.com/tensorflow/io/actions?query=branch%3Amaster)
[![PyPI](https://badge.fury.io/py/tensorflow-io.svg)](https://pypi.org/project/tensorflow-io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/tensorflow/io/blob/master/LICENSE)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/io)

TensorFlow I/O is a collection of file systems and file formats that are not
available in TensorFlow's built-in support. A full list of supported file systems
and file formats by TensorFlow I/O can be found [here](https://www.tensorflow.org/io/api_docs/python/tfio).

The use of tensorflow-io is straightforward with keras. Below is an example
to [Get Started with TensorFlow](https://www.tensorflow.org/tutorials/quickstart/beginner) with
the data processing aspect replaced by tensorflow-io:

```python
import tensorflow as tf
import tensorflow_io as tfio

# Read the MNIST data into the IODataset.
dataset_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
d_train = tfio.IODataset.from_mnist(
    dataset_url + "train-images-idx3-ubyte.gz",
    dataset_url + "train-labels-idx1-ubyte.gz",
)

# Shuffle the elements of the dataset.
d_train = d_train.shuffle(buffer_size=1024)

# By default image data is uint8, so convert to float32 using map().
d_train = d_train.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y))

# prepare batches the data just like any other tf.data.Dataset
d_train = d_train.batch(32)

# Build the model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model.
model.fit(d_train, epochs=5, steps_per_epoch=200)
```

In the above [MNIST](http://yann.lecun.com/exdb/mnist/) example, the URL's
to access the dataset files are passed directly to the `tfio.IODataset.from_mnist` API call.
This is due to the inherent support that `tensorflow-io` provides for `HTTP`/`HTTPS` file system,
thus eliminating the need for downloading and saving datasets on a local directory.

NOTE: Since `tensorflow-io` is able to detect and uncompress the MNIST dataset automatically if needed,
we can pass the URL's for the compressed files (gzip) to the API call as is.

Please check the official [documentation](https://www.tensorflow.org/io) for more
detailed and interesting usages of the package.

## Installation

### Python Package

The `tensorflow-io` Python package can be installed with pip directly using:
```sh
$ pip install tensorflow-io
```

People who are a little more adventurous can also try our nightly binaries:
```sh
$ pip install tensorflow-io-nightly
```

In addition to the pip packages, the docker images can be used to quickly get started.

For stable builds:
```sh
$ docker pull tfsigio/tfio:latest
$ docker run -it --rm --name tfio-latest tfsigio/tfio:latest
```

For nightly builds:
```sh
$ docker pull tfsigio/tfio:nightly
$ docker run -it --rm --name tfio-nightly tfsigio/tfio:nightly
```

### R Package

Once the `tensorflow-io` Python package has been successfully installed, you
can install the development version of the R package from GitHub via the following:
```r
if (!require("remotes")) install.packages("remotes")
remotes::install_github("tensorflow/io", subdir = "R-package")
```

### TensorFlow Version Compatibility

To ensure compatibility with TensorFlow, it is recommended to install a matching
version of TensorFlow I/O according to the table below. You can find the list
of releases [here](https://github.com/tensorflow/io/releases).

| TensorFlow I/O Version | TensorFlow Compatibility | Release Date |
| --- | --- | --- |
| 0.17.0 | 2.4.x | Dec 14, 2020 |
| 0.16.0 | 2.3.x | Oct 23, 2020 |
| 0.15.0 | 2.3.x | Aug 03, 2020 |
| 0.14.0 | 2.2.x | Jul 08, 2020 |
| 0.13.0 | 2.2.x | May 10, 2020 |
| 0.12.0 | 2.1.x | Feb 28, 2020 |
| 0.11.0 | 2.1.x | Jan 10, 2020 |
| 0.10.0 | 2.0.x | Dec 05, 2019 |
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

## Development

### IDE Setup

For instructions on how to configure Visual Studio Code for developing TensorFlow I/O, please refer to
https://github.com/tensorflow/io/blob/master/docs/vscode.md

### Lint

TensorFlow I/O's code conforms to Bazel Buildifier, Clang Format, Black, and Pyupgrade.
Please use the following command to check the source code and identify lint issues:
```
$ bazel run //tools/lint:check
```

For Bazel Buildifier and Clang Format, the following command will automatically identify
and fix any lint errors:
```
$ bazel run //tools/lint:lint
```

Alternatively, if you only want to perform lint check using individual linters,
then you can selectively pass `black`, `pyupgrade`, `bazel`, or `clang` to the above commands.

For example, a `black` specific lint check can be done using:
```
$ bazel run //tools/lint:check -- black
```

Lint fix using Bazel Buildifier and Clang Format can be done using:
```
$ bazel run //tools/lint:lint -- bazel clang
```

Lint check using `black` and `pyupgrade` for an individual python file can be done using:
```
$ bazel run //tools/lint:check -- black pyupgrade -- tensorflow_io/core/python/ops/version_ops.py
```

Lint fix an individual python file with black and pyupgrade using:
```
$ bazel run //tools/lint:lint -- black pyupgrade --  tensorflow_io/core/python/ops/version_ops.py
```


### Python

#### macOS

On macOS Catalina 10.15.7, it is possible to build tensorflow-io with
system provided python 3.8.2. Both `tensorflow` and `bazel` are needed.

NOTE: The system default python 3.8.2 on macOS 10.15.7 will cause `regex` installation
error caused by compiler option of `-arch arm64 -arch x86_64` (similar to the issue
mentioned in https://github.com/giampaolo/psutil/issues/1832). To overcome this issue
`export ARCHFLAGS="-arch x86_64"` will be needed to remove arm64 build option.

```sh
#!/usr/bin/env bash

# Disable arm64 build by specifying only x86_64 arch.
# Only needed for macOS's system default python 3.8.2 on macOS 10.15.7
export ARCHFLAGS="-arch x86_64"

# Use following command to check if Xcode is correctly installed:
xcodebuild -version

# Show macOS's default python3
python3 --version

# Install Bazel version specified in .bazelversion
curl -OL https://github.com/bazelbuild/bazel/releases/download/$(cat .bazelversion)/bazel-$(cat .bazelversion)-installer-darwin-x86_64.sh
sudo bash -x -e bazel-$(cat .bazelversion)-installer-darwin-x86_64.sh

# Install tensorflow and configure bazel
sudo ./configure.sh

# Build shared libraries
bazel build -s --verbose_failures //tensorflow_io/...

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core/python/ops/` and it is possible
# to run tests with `pytest`, e.g.:
sudo python3 -m pip install pytest
TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_serialization_eager.py
```

NOTE: When running pytest, `TFIO_DATAPATH=bazel-bin` has to be passed so that python can utilize the generated shared libraries after the build process.

##### Troubleshoot

If Xcode is installed, but `$ xcodebuild -version` is not displaying the expected output, you might need to enable Xcode command line with the command:

`$ xcode-select -s /Applications/Xcode.app/Contents/Developer`.

A terminal restart might be required for the changes to take effect.

Sample output:

```
$ xcodebuild -version
Xcode 11.6
Build version 11E708
```


#### Linux

Development of tensorflow-io on Linux is similar to macOS. The required packages
are gcc, g++, git, bazel, and python 3. Newer versions of gcc or python, other than the default system installed
versions might be required though.

##### Ubuntu 20.04

Ubuntu 20.04 requires gcc/g++, git, and python 3. The following will install dependencies and build
the shared libraries on Ubuntu 20.04:
```sh
#!/usr/bin/env bash

# Install gcc/g++, git, unzip/curl (for bazel), and python3
sudo apt-get -y -qq update
sudo apt-get -y -qq install gcc g++ git unzip curl python3-pip

# Install Bazel version specified in .bazelversion
curl -sSOL https://github.com/bazelbuild/bazel/releases/download/$(cat .bazelversion)/bazel-$(cat .bazelversion)-installer-linux-x86_64.sh
sudo bash -x -e bazel-$(cat .bazelversion)-installer-linux-x86_64.sh

# Upgrade pip
sudo python3 -m pip install -U pip

# Install tensorflow and configure bazel
sudo ./configure.sh

# Build shared libraries
bazel build -s --verbose_failures //tensorflow_io/...

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core/python/ops/` and it is possible
# to run tests with `pytest`, e.g.:
sudo python3 -m pip install pytest
TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_serialization_eager.py
```

##### CentOS 8

CentOS 8 requires gcc/g++, git, and python 3. The following will install dependencies and build
the shared libraries on CentOS 8:
```sh
#!/usr/bin/env bash

# Install gcc/g++, git, unzip/which (for bazel), and python3
sudo yum install -y python3 python3-devel gcc gcc-c++ git unzip which make

# Install Bazel version specified in .bazelversion
curl -sSOL https://github.com/bazelbuild/bazel/releases/download/$(cat .bazelversion)/bazel-$(cat .bazelversion)-installer-linux-x86_64.sh
sudo bash -x -e bazel-$(cat .bazelversion)-installer-linux-x86_64.sh

# Upgrade pip
sudo python3 -m pip install -U pip

# Install tensorflow and configure bazel
sudo ./configure.sh

# Build shared libraries
bazel build -s --verbose_failures //tensorflow_io/...

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core/python/ops/` and it is possible
# to run tests with `pytest`, e.g.:
sudo python3 -m pip install pytest
TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_serialization_eager.py
```

##### CentOS 7

On CentOS 7, the default python and gcc version are too old to build tensorflow-io's shared
libraries (.so). The gcc provided by Developer Toolset and rh-python36 should be used instead.
Also, the libstdc++ has to be linked statically to avoid discrepancy of libstdc++ installed on
CentOS vs. newer gcc version by devtoolset.

Furthermore, a special flag `--//tensorflow_io/core:static_build` has to be passed to Bazel
in order to avoid duplication of symbols in statically linked libraries for file system
plugins.

The following will install bazel, devtoolset-9, rh-python36, and build the shared libraries:
```sh
#!/usr/bin/env bash

# Install centos-release-scl, then install gcc/g++ (devtoolset), git, and python 3
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-9 git rh-python36 make

# Install Bazel version specified in .bazelversion
curl -sSOL https://github.com/bazelbuild/bazel/releases/download/$(cat .bazelversion)/bazel-$(cat .bazelversion)-installer-linux-x86_64.sh
sudo bash -x -e bazel-$(cat .bazelversion)-installer-linux-x86_64.sh

# Upgrade pip
scl enable rh-python36 devtoolset-9 \
    'python3 -m pip install -U pip'

# Install tensorflow and configure bazel with rh-python36
scl enable rh-python36 devtoolset-9 \
    './configure.sh'

# Build shared libraries, notice the passing of --//tensorflow_io/core:static_build
BAZEL_LINKOPTS="-static-libstdc++ -static-libgcc" BAZEL_LINKLIBS="-lm -l%:libstdc++.a" \
  scl enable rh-python36 devtoolset-9 \
    'bazel build -s --verbose_failures --//tensorflow_io/core:static_build //tensorflow_io/...'

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core/python/ops/` and it is possible
# to run tests with `pytest`, e.g.:
scl enable rh-python36 devtoolset-9 \
    'python3 -m pip install pytest'

TFIO_DATAPATH=bazel-bin \
  scl enable rh-python36 devtoolset-9 \
    'python3 -m pytest -s -v tests/test_serialization_eager.py'
```

#### Python Wheels

It is possible to build python wheels after bazel build is complete with the following command:
```
$ python3 setup.py bdist_wheel --data bazel-bin
```
The .whl file will be available in dist directory. Note the bazel binary directory `bazel-bin`
has to be passed with `--data` args in order for setup.py to locate the necessary share objects,
as `bazel-bin` is outside of the `tensorflow_io` package directory.

Alternatively, source install could be done with:
```
$ TFIO_DATAPATH=bazel-bin python3 -m pip install .
```
with `TFIO_DATAPATH=bazel-bin` passed for the same reason.

Note installing with `-e` is different from the above. The
```
$ TFIO_DATAPATH=bazel-bin python3 -m pip install -e .
```
will not install shared object automatically even with `TFIO_DATAPATH=bazel-bin`. Instead,
`TFIO_DATAPATH=bazel-bin` has to be passed everytime the program is run after the install:
```
$ TFIO_DATAPATH=bazel-bin python3

>>> import tensorflow_io as tfio
>>> ...
```

#### Docker

For Python development, a reference Dockerfile [here](tools/docker/devel.Dockerfile) can be
used to build the TensorFlow I/O package (`tensorflow-io`) from source. Additionally, the
pre-built devel images can be used as well:
```sh
# Pull (if necessary) and start the devel container
$ docker run -it --rm --name tfio-dev --net=host -v ${PWD}:/v -w /v tfsigio/tfio:latest-devel bash

# Inside the docker container, ./configure.sh will install TensorFlow or use existing install
(tfio-dev) root@docker-desktop:/v$ ./configure.sh

# Clean up exisiting bazel build's (if any)
(tfio-dev) root@docker-desktop:/v$ rm -rf bazel-*

# Build TensorFlow I/O C++. For compilation optimization flags, the default (-march=native)
# optimizes the generated code for your machine's CPU type.
# Reference: https://www.tensorflow.orginstall/source#configuration_options).

# NOTE: Based on the available resources, please change the number of job workers to:
# -j 4/8/16 to prevent bazel server terminations and resource oriented build errors.

(tfio-dev) root@docker-desktop:/v$ bazel build -j 8 --copt=-msse4.2 --copt=-mavx --compilation_mode=opt --verbose_failures --test_output=errors --crosstool_top=//third_party/toolchains/gcc7_manylinux2010:toolchain //tensorflow_io/...


# Run tests with PyTest, note: some tests require launching additional containers to run (see below)
(tfio-dev) root@docker-desktop:/v$ pytest -s -v tests/
# Build the TensorFlow I/O package
(tfio-dev) root@docker-desktop:/v$ python setup.py bdist_wheel
```

A package file `dist/tensorflow_io-*.whl` will be generated after a build is successful.

NOTE: When working in the Python development container, an environment variable
`TFIO_DATAPATH` is automatically set to point tensorflow-io to the shared C++
libraries built by Bazel to run `pytest` and build the `bdist_wheel`. Python
`setup.py` can also accept `--data [path]` as an argument, for example
`python setup.py --data bazel-bin bdist_wheel`.

NOTE: While the tfio-dev container gives developers an easy to work with
environment, the released whl packages are built differently due to manylinux2010
requirements. Please check [Build Status and CI] section for more details
on how the released whl packages are generated.

#### Starting Test Containers

Some tests require launching a test container before running. In order
to run all tests, execute the following commands:

```sh
$ bash -x -e tests/test_ignite/start_ignite.sh
$ bash -x -e tests/test_kafka/kafka_test.sh
$ bash -x -e tests/test_kinesis/kinesis_test.sh
```

### R

We provide a reference Dockerfile [here](R-package/scripts/Dockerfile) for you
so that you can use the R package directly for testing. You can build it via:
```sh
$ docker build -t tfio-r-dev -f R-package/scripts/Dockerfile .
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
#!/usr/bin/env bash

ls dist/*
for f in dist/*.whl; do
  docker run -i --rm -v $PWD:/v -w /v --net=host quay.io/pypa/manylinux2010_x86_64 bash -x -e /v/tools/build/auditwheel repair --plat manylinux2010_x86_64 $f
done
sudo chown -R $(id -nu):$(id -ng) .
ls wheelhouse/*
```

It takes some time to build, but once complete, there will be python
`3.5`, `3.6`, `3.7` compatible whl packages available in `wheelhouse`
directory.

On macOS, the same command could be used though the script expect `python` in shell
and will only generate a whl package that matches the version of `python` in shell. If
you want to build a whl package for a specific python then you have to alias this version
of python to `python` in shell. See [.github/workflows/build.yml](.github/workflows/build.yml)
Auditwheel step for instructions how to do that.

Note the above command is also the command we use when releasing packages for Linux and macOS.

TensorFlow I/O uses both GitHub Workflows and Google CI (Kokoro) for continuous integration.
GitHub Workflows is used for macOS build and test. Kokoro is used for Linux build and test.
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
are done with live systems, meaning we install Prometheus/Kafka/Ignite on CI machine before
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
- Official [PubSub Emulator](https://cloud.google.com/sdk/gcloud/reference/beta/emulators/pubsub/) by Google Cloud for Cloud PubSub.
- Official [Azurite Emulator](https://github.com/Azure/Azurite) by Azure for Azure Storage.
- None-official [LocalStack emulator](https://github.com/localstack/localstack) by LocalStack for AWS Kinesis.


## Community

* SIG IO [Google Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/io) and mailing list: [io@tensorflow.org](io@tensorflow.org)
* SIG IO [Monthly Meeting Notes](https://docs.google.com/document/d/1CB51yJxns5WA4Ylv89D-a5qReiGTC0GYum6DU-9nKGo/edit)
* Gitter room: [tensorflow/sig-io](https://gitter.im/tensorflow/sig-io)

## More Information

* [Streaming Machine Learning with Tiered Storage and Without a Data Lake](https://www.confluent.io/blog/streaming-machine-learning-with-tiered-storage/) - [Kai Waehner](https://github.com/kaiwaehner)
* [TensorFlow with Apache Arrow Datasets](https://medium.com/tensorflow/tensorflow-with-apache-arrow-datasets-cdbcfe80a59f) - [Bryan Cutler](https://github.com/BryanCutler)
* [How to build a custom Dataset for Tensorflow](https://towardsdatascience.com/how-to-build-a-custom-dataset-for-tensorflow-1fe3967544d8) - [Ivelin Ivanov](https://github.com/ivelin)
* [TensorFlow on Apache Ignite](https://medium.com/tensorflow/tensorflow-on-apache-ignite-99f1fc60efeb) - [Anton Dmitriev](https://github.com/dmitrievanthony)

## License

[Apache License 2.0](LICENSE)
