# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorGenerator"""

import hashlib
import numpy as np
import tensorflow as tf

from tests.test_atds_avro.utils.generator.generator_base import (
    Generator,
)
from tests.test_atds_avro.utils.hash_util import int_to_bytes


class TensorGeneratorBase(Generator):
    """Base of TensorGenerator that generates tf.Tensor."""

    def __init__(self, spec):
        """Create a new TensorGeneratorBase.

        This must be called by the constructors of subclasses e.g.
        IntTensorGenerator, FloatTensorGenerator, etc.

        Args:
          spec: A tf.TensorSpec that describes the output tensor.

        Raises:
          TypeError: If spec is not tf.TensorSpec.
          ValueError: If shape in spec is not fully defined.
        """
        super().__init__(spec)

        if not isinstance(spec, tf.TensorSpec):
            raise TypeError(
                "Input spec must be a tf.TensorSpec in TensorGenerator "
                f"but found {spec}"
            )

        spec.shape.assert_is_fully_defined()


class IntTensorGenerator(TensorGeneratorBase):
    """IntTensorGenerator generates tf.Tensor with dtype in tf.int32 or tf.int64"""

    def __init__(self, spec):
        """Create a new IntTensorGenerator

        With tf.int32 dtype, the generated int range is between -2^31 to 2^31 - 1.
        With tf.int64 dtype, the generated int range is between -2^63 to 2^63 - 1.

        Args:
          spec: A tf.TensorSpec that describes the output tensor.

        Raises:
          TypeError: If dtype in spec is not tf.int32 or tf.int64.
        """
        super().__init__(spec)

        if spec.dtype not in [tf.int32, tf.int64]:
            raise TypeError(
                "IntTensorGenerator can only generate tf.Tensor with "
                f"dtype in tf.int32 or tf.int64 but found {spec.dtype}."
            )

    def generate(self):
        dtype = self.spec.dtype
        info = np.iinfo(dtype.as_numpy_dtype)
        shape = self.spec.shape.as_list()
        values = np.random.randint(
            low=info.min, high=info.max, size=shape, dtype=dtype.as_numpy_dtype
        )
        return tf.convert_to_tensor(values, dtype=dtype, name=self.spec.name)


class FloatTensorGenerator(TensorGeneratorBase):
    """FloatTensorGenerator generates tf.Tensor with dtype in tf.float32
    or tf.float64."""

    def __init__(self, spec):
        """Create a new FloatTensorGenerator

        The generated float range is between 0.0 to 1.0.

        Args:
          spec: A tf.TensorSpec that describes the output tensor.

        Raises:
          TypeError: If dtype in spec is not tf.float32 or tf.float64.
        """
        super().__init__(spec)

        if spec.dtype not in [tf.float32, tf.float64]:
            raise TypeError(
                "FloatTensorGenerator can only generate tf.Tensor with "
                f"dtype in tf.float32 or tf.float64 but found {spec.dtype}."
            )

    def generate(self):
        shape = self.spec.shape.as_list()
        values = np.random.rand(*shape)
        return tf.convert_to_tensor(values, dtype=self.spec.dtype, name=self.spec.name)


class WordTensorGenerator(TensorGeneratorBase):
    """WordTensorGenerator generates string tf.Tensor with string
    length similar to a word."""

    def __init__(self, spec, avg_length=5):
        """Create a new WordTensorGenerator

        WordTensorGenerator samples word length using Poisson distribution
        with lambda equals to avg_length and generates random bytes for each word.

        Args:
          spec: A tf.TensorSpec that describes the output tensor.
          avg_length: An int that represents the average word length.

        Raises:
          TypeError: If dtype in spec is not tf.string.
          ValueError: If avg_length is not positive.
        """
        super().__init__(spec)

        if spec.dtype is not tf.string:
            raise TypeError(
                "WordTensorGenerator can only generate tf.Tensor with "
                f"dtype in tf.string but found {spec.dtype}."
            )

        if avg_length < 1:
            raise ValueError(
                "WordTensorGenerator must have positive avg_length"
                f" but found {avg_length}."
            )
        self._avg_length = avg_length

    def generate(self):
        # Use Poisson distribution to sample the length of byte strings.
        # The avg_length equals to the lambda in Poisson distribution.
        shape = self.spec.shape.as_list()
        lengths = np.random.poisson(self._avg_length, size=shape)

        to_string = lambda length: np.random.bytes(length)
        vfunc = np.vectorize(to_string)
        values = vfunc(lengths)

        return tf.convert_to_tensor(values, dtype=tf.string, name=self.spec.name)

    def hash_code(self):
        """Return the hashed code of this Generator in hex str."""
        hash_code = super().hash_code()

        m = hashlib.sha256()
        m.update(hash_code.encode())
        m.update(int_to_bytes(self._avg_length))
        return m.hexdigest()


class BoolTensorGenerator(TensorGeneratorBase):
    """BoolTensorGenerator generates tf.Tensor with dtype in tf.bool."""

    def __init__(self, spec):
        """Create a new BoolTensorGenerator.

        The generated bool value has equal true and false possibility.

        Args:
          spec: A tf.TensorSpec that describes the output tensor.

        Raises:
          TypeError: If dtype in spec is not tf.bool.
        """
        super().__init__(spec)

        if spec.dtype is not tf.bool:
            raise TypeError(
                "BoolTensorGenerator can only generate tf.Tensor with "
                f"dtype in tf.bool but found {spec.dtype}."
            )

    def generate(self):
        shape = self.spec.shape.as_list()
        # np.random.rand generates values from 0 to 1 using Uniform distribution
        values = np.random.rand(*shape) > 0.5
        return tf.convert_to_tensor(values, dtype=tf.bool, name=self.spec.name)
