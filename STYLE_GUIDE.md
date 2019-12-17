# Coding guidelines and standards

## Coding Style

#### C++ 

C++ code should conform to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).


#### Python

Python code should conform to [PEP8](https://www.python.org/dev/peps/pep-0008/).

##### Running Python and Bazel Style Checks

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


## Naming Conventions

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