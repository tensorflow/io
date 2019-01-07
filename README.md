# TensorFlow I/O

[![Travis-CI Build Status](https://travis-ci.org/tensorflow/io.svg?branch=master)](https://travis-ci.org/tensorflow/io) 

TensorFlow I/O is a collection of file systems and file formats that are not
available in TensorFlow's built-in support.

At the moment TensorFlow I/O supports 4 data sources:
- `tensorflow_io.ignite`: Data source for Apache Ignite and Ignite File System (IGFS).
- `tensorflow_io.kafka`: Apache Kafka stream-processing support.
- `tensorflow_io.kinesis`: Amazon Kinesis data streams support.
- `tensorflow_io.hadoop`: Hadoop SequenceFile format support.

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

## Developing

### Python

The TensorFlow I/O package (`tensorflow-io`) could be built from source:
```sh
$ docker run -it -v ${PWD}:/working_dir -w /working_dir tensorflow/tensorflow:custom-op
$ # In docker
$ ./configure.sh
$ bazel build build_pip_pkg
$ bazel-bin/build_pip_pkg artifacts
```

A package file `artifacts/tensorflow_io-*.whl` will be generated after a build is successful.

### R

The docker image `tensorflow/tensorflow:custom-op` does not have R installation.
Instead, the latest R package is available from
[CRAN](https://cran.r-project.org/bin/linux/ubuntu/README.html).

After the successful installation of `R`(`r-base`) from CRAN, the following installation steps
are also needed:
```
$ # In docker
$ apt-get -y -qq update
$ apt-get -y -qq install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev
$ R -e 'install.packages(c("devtools", "testthat", "tensorflow", "tfdatasets"), dependencies = TRUE)'
```

With all prerequisites packages installed, it is possible to run test with:
```
$ # In docker
$ R -e "devtools::test()"
```

Alternatively, we also provided a reference Dockerfile [here](R-package/scripts/Dockerfile) for you
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
```

## License

[Apache License 2.0](LICENSE)
