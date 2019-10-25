<div align="center">
  <img src="https://github.com/tensorflow/community/blob/master/sigs/logos/SIGIO.png" width="60%"><br><br>
</div>

-----------------

# Contributing

Tensorflow IO project welcomes all kinds of contributions, be it code changes, bug-fixes or documentation changes.
This guide should help you in taking care of some basic setups & code conventions.
 
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

## Coding guidelines and standards

### Naming Conventions

Any new operation that is introduced in Tensorflow I/O project should be prefixed with a pre-defined text.
This is done to ensure that different Tensorflow projects (for e.g Tensorflow I/O) declare their operation names that are globally unique,
across the entire Tensorflow ecosystem. 

For more details on defining custom operations, please refer this [tensorflow RFC](https://github.com/tensorflow/community/pull/126).

Depending on the module's programming language (C++ or python), pre-defined prefix text may vary. This is detailed below.
 
#### C++

Any new operation that is introduced in C++ module should be prefixed with text “IO>”. 

for example:

A new operation named “ReadAvro” should be registered as “IO>ReadAvro”.

##### Regular way:

```text
REGISTER_OP("ReadAvro")
                    ...
                    ...
                    ...
     });
```

##### Suggested way:

```text
REGISTER_OP("IO>ReadAvro")
                    ...
                    ...
                    ...
     });
```

Please note in suggested way, how the operation name is prefixed with "IO>".

#### Python 

Any new operation that is introduced in python module and that which corresponds to the operation in C++ module,
should be prefixed with "io_". This ensures that C++ operation binding is mapped correctly.

for example:

The python equivalent declaration of above operation named “IO>ReadAvro” would be "io_read_avro".

```text
def read_avro(filename, schema, column, **kwargs):
  """read_avro"""
  ...
  ...
  ...
  return core_ops.io_read_avro(...);
```

## Code Reviews

All submissions, including submissions by project members, require review. We
use Github pull requests for this purpose.
Tensorflow IO project's currently open pull requests, 
can be viewed [here](https://github.com/tensorflow/io/pulls).

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to https://cla.developers.google.com/ to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.