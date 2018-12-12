# TensorFlow I/O

[![Travis-CI Build Status](https://travis-ci.org/tensorflow/io.svg?branch=master)](https://travis-ci.org/tensorflow/io) 

TensorFlow I/O is a collection of file systems and file formats that are not
available in TensorFlow's built-in support.

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

## Installation

Once a package file `artifacts/tensorflow_io-*.whl` is ready, installation could be done through:
```
$ pip install working_dir/artifacts/tensorflow_io-*.whl
```

The related module could be imported with python:
```
$  python
Python 2.7.6 (default, Nov 13 2018, 12:45:42)
[GCC 4.8.4] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> from tensorflow_io.kafka import KafkaDataset
>>>
```

Note that python has to run outside of repo directory itself, otherwise python may not
be able to find the correct path to the module.

## License

[Apache License 2.0](LICENSE)
