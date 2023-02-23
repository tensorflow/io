# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""MockGenerator"""

import tensorflow as tf

from tensorflow_io.python.experimental.benchmark.generator.generator_base \
  import Generator


class MockGenerator(Generator):
  """MockGenerator is a test utility class that generates tensor based on
   the given data."""

  def __init__(self, spec, data, generator_cls):
    """Create a new MockGenerator

    MockGenerator generates tensors by returning tensors from the given data.

    Args:
      spec: A tf.TensorSpec that describes the output tensor.
      data: A list of tensor to generate.
      generator_cls: Class of wrapped generator object.

    Raises:
      ValueError: If spec is not compatible with data or data is empty.
    """
    super(MockGenerator, self).__init__(spec)

    for index, tensor in enumerate(data):
      if not spec.is_compatible_with(tensor):
        raise ValueError("Input spec and data are not compatible."
                         f"Tensor {tensor} at {index}th location is not "
                         f"compatible with input spec {spec}")
    if not data:
      raise ValueError("Input data should not be empty.")

    self._data = data
    self._index = 0
    self._generator_cls = generator_cls

  def generate(self):
    """Generate output tensor by returning tensors from input data.
    Output tensor was returned based on its order in input data.
    When all tensors are returned, this generator will restart from beginning.
    """
    if self._index >= len(self._data):
      self._index = 0

    tensor = self._data[self._index]
    self._index += 1
    return tensor

  def get_generator_cls(self):
    """Get the generator class which this object represents."""
    return self._generator_cls
