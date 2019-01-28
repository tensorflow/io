# TensorFlow I/O

[![Travis-CI Build Status](https://travis-ci.org/tensorflow/io.svg?branch=master)](https://travis-ci.org/tensorflow/io)
[![PyPI Status Badge](https://badge.fury.io/py/tensorflow-io.svg)](https://pypi.org/project/tensorflow-io/)
[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/tfio)](https://cran.r-project.org/package=tfio)

TensorFlow I/O is a collection of file systems and file formats that are not
available in TensorFlow's built-in support.

At the moment TensorFlow I/O supports 5 data sources:
- `tensorflow_io.ignite`: Data source for Apache Ignite and Ignite File System (IGFS). Overview and usage guide [here](tensorflow_io/ignite/README.md).
- `tensorflow_io.kafka`: Apache Kafka stream-processing support.
- `tensorflow_io.kinesis`: Amazon Kinesis data streams support.
- `tensorflow_io.hadoop`: Hadoop SequenceFile format support.
- `tensorflow_io.arrow`: Apache Arrow data format support. Usage guide [here](tensorflow_io/arrow/README.md).
- `tensorflow_io.image`: WebP image format support.
- `tensorflow_io.libsvm`: LIBSVM file format support.
- `tensorflow_io.video`: Video file support with FFmpeg.

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
