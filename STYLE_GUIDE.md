## Coding guidelines and standards

This document is a reference guide for contributing new ops to the tensorflow-io project.

### Coding Style

- C++ code should conform to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
- Python code should conform to [PEP8](https://www.python.org/dev/peps/pep-0008/).


Style checks for C++, Python and Bazel can be automatically rectified with the following command:
```
$ bazel run //tools/lint:lint
```

### Naming Conventions

Any new op that is introduced in Tensorflow I/O should have a pre-defined prefix. Adherence to this guideline ensures that different Tensorflow projects (for e.g Tensorflow I/O) declare their op names that are globally unique, across the entire Tensorflow ecosystem.

- For details on writing custom ops, please refer this [guide](https://www.tensorflow.org/guide/create_op).
- For more details on naming custom ops, please refer this [tensorflow RFC](https://github.com/tensorflow/community/pull/126).

Depending on the module's programming language (C++ or python), the pre-defined prefix may vary.
#### C++

The C++ based ops are placed in `tensorflow_io/core/ops` and are prefixed with text `IO>`
while registering. For instance, a new op named `AudioResample` should be registered as
`IO>AudioResample`.

```cc
// tensorflow_io/core/ops/audio_ops.cc
...
REGISTER_OP("IO>AudioResample")
    .Input("input: T")
    .Input("rate_in: int64")
    .Input("rate_out: int64")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
      return Status::OK();
    });
```
_Observe how the op name is prefixed with `IO>`_

The next step would be to write the kernels for this newly defined op. The kernel implementations are placed in `tensorflow_io/core/kernels` and implement the business logic for the op. Additional details can be found in this [guide](https://www.tensorflow.org/guide/create_op).
#### Python

Any new op that is introduced in a python module and that which corresponds to an op in C++, must be prefixed with `io_`. This ensures that the C++ op is properly bound to this python target.
For example, the python equivalent declaration of above mentioned C++ op named `IO>AudioResample` would be `io_audio_resample`.

```python
from tensorflow_io.python.ops import core_ops

def resample(input, rate_in, rate_out, name=None):
  """resample audio"""

  # implementation logic#

  return core_ops.io_audio_resample(...)
```

Few things to note:
- The C++ op name is converted to lowercase and is `_` delimited i.e, `io_audio_resample` is now bound to `IO>AudioResample`.
- The `io_audio_resample` target is available in `core_ops` This import hierarchy is due to the nature of the build targets defined [here](https://github.com/tensorflow/io/blob/master/tensorflow_io/BUILD)