# TensorFlow I/O

[![Travis-CI Build Status](https://travis-ci.org/tensorflow/io.svg?branch=master)](https://travis-ci.org/tensorflow/io) 

TensorFlow I/O is a collection of file systems and file formats that are not
available in TensorFlow's built-in support.

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

The TensorFlow I/O package (`tensorflow-io`) could be built from source:
```sh
$ docker run -it -v ${PWD}:/working_dir -w /working_dir  tensorflow/tensorflow:custom-op
$ # In docker
$ ./configure.sh
$ bazel build build_pip_pkg
$ bazel-bin/build_pip_pkg artifacts
```

A package file `artifacts/tensorflow_io-*.whl` will be generated after a build is successful.

## License

[Apache License 2.0](LICENSE)
