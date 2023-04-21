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
"""SparseTensorGenerator"""

import enum
import random
import hashlib
import numpy as np
import tensorflow as tf

from tests.test_atds_avro.utils.generator.generator_base import (
    Generator,
)
from tests.test_atds_avro.utils.hash_util import int_to_bytes


class ValueDistribution(enum.Enum):
    SINGLE_VALUE = 1
    SMALL_NUM_VALUE = 2  # 5 to 9 elements
    LARGE_NUM_VALUE = 3  # 100 to 999 elements


_VALUE_DISTRIBUTION_TO_RANGE = {
    ValueDistribution.SINGLE_VALUE: (1, 2),
    ValueDistribution.SMALL_NUM_VALUE: (5, 10),
    ValueDistribution.LARGE_NUM_VALUE: (100, 1000),
}


def get_common_value_dist():
    # Assume tensor is one-hot since this is a common use case
    return ValueDistribution.SINGLE_VALUE


class SparseTensorGeneratorBase(Generator):
    """Base of SparseTensorGenerator that generates tf.sparse.SparseTensor."""

    def __init__(self, spec, num_values):
        """Create a new SparseTensorGeneratorBase.

        This must be called by the constructors of subclasses e.g.
        IntSparseTensorGenerator, FloatSparseTensorGenerator, etc.

        Args:
          spec: A tf.SparseTensorSpec that describes the output tensor.
          num_values: A value distribution or an int specifying number of non-zero values in the sparse tensor.

        Raises:
          TypeError: If spec is not tf.SparseTensorSpec.
        """
        super().__init__(spec)

        if not isinstance(spec, tf.SparseTensorSpec):
            raise TypeError(
                "Input spec must be a tf.SparseTensorSpec in SparseTensorGenerator "
                f"but found {spec}"
            )

        if not isinstance(num_values, (int, ValueDistribution)):
            raise TypeError(
                f"num_values must be an int or ValueDistribution but found {num_values}"
            )

        self._value_dist = None
        if isinstance(num_values, int):
            self._num_values = num_values
        elif num_values == ValueDistribution.SINGLE_VALUE:
            self._num_values = 1
        else:
            self._value_dist = num_values

    def _get_num_values(self, shape):
        ret = 0
        if self._value_dist == None:
            ret = self._num_values
        else:
            ret = np.random.randint(*_VALUE_DISTRIBUTION_TO_RANGE[self._value_dist])
        return min(ret, np.prod(shape))

    def _get_shape(self):
        # If spec shape is None, generate shape with random rank between 1 and 5
        shape = (
            [None] * np.random.randint(1, 5)
            if self.spec.shape == None
            else self.spec.shape
        )
        # Populate unknown dimensions with random int between 1 and 10
        return [dim if dim != None else np.random.randint(1, 10) for dim in shape]

    def _generate_random_coords(self, num_values, shape):
        if num_values == 0:
            return np.empty((0, len(shape)), dtype=np.int64)
        indices = sorted(random.sample(range(np.prod(shape)), num_values))
        return [self._int_to_coord(idx, shape) for idx in indices]

    def _int_to_coord(self, idx, shape):
        """Convert an integer to its corresponding location in a tensor, in row-major order.
        For example, in the 2d tensor
        [[0, 1, 2]
         [3, 4, 5]]
        The index 3 will return [1, 0] (i.e. the entry in the second row, first column)
        """
        rank = len(shape)
        ret = [0] * rank
        for dim in range(rank):
            val = idx % (shape[rank - dim - 1])
            ret[rank - dim - 1] = val
            idx = (idx - val) // shape[rank - dim - 1]
        return ret

    def hash_code(self):
        hash_code = super().hash_code()

        m = hashlib.sha256()
        m.update(hash_code.encode())

        # Hash input num_values
        if self._value_dist:
            m.update(ValueDistribution.__name__.encode())
            m.update(self._value_dist.name.encode())  # num_values is enum
        else:
            m.update(int_to_bytes(self._num_values))  # num_values is constant int.
        return m.hexdigest()


class IntSparseTensorGenerator(SparseTensorGeneratorBase):
    """IntSparseTensorGenerator generates tf.sparse.SparseTensor with dtype in tf.int32 or tf.int64"""

    def __init__(self, spec, num_values):
        """Create a new IntSparseTensorGenerator

        With tf.int32 dtype, the generated int range is between -2^31 to 2^31 - 1.
        With tf.int64 dtype, the generated int range is between -2^63 to 2^63 - 1.

        Args:
          spec: A tf.SparseTensorSpec that describes the output tensor.
          num_values: A value distribution or an int specifying number of non-zero values in the sparse tensor.

        Raises:
          TypeError: If dtype in spec is not tf.int32 or tf.int64.
        """
        super().__init__(spec, num_values)

        if spec.dtype not in [tf.int32, tf.int64]:
            raise TypeError(
                "IntSparseTensorGenerator can only generate tf.sparse.SparseTensor with "
                f"dtype in tf.int32 or tf.int64 but found {spec.dtype}."
            )

    def generate(self):
        dtype = self.spec.dtype
        info = np.iinfo(dtype.as_numpy_dtype)
        shape = self._get_shape()
        num_values = self._get_num_values(shape)
        vals = np.random.randint(
            low=info.min, high=info.max, size=[num_values], dtype=dtype.as_numpy_dtype
        )
        coords = self._generate_random_coords(num_values, shape)
        return tf.SparseTensor(indices=coords, values=vals, dense_shape=shape)


class FloatSparseTensorGenerator(SparseTensorGeneratorBase):
    """FloatSparseTensorGenerator generates tf.sparse.SparseTensor with dtype in tf.float32
    or tf.float64."""

    def __init__(self, spec, num_values):
        """Create a new FloatSparseTensorGenerator

        The generated float range is between 0.0 to 1.0.

        Args:
          spec: A tf.SparseTensorSpec that describes the output tensor.
          num_values: A value distribution or an int specifying number of non-zero values in the sparse tensor.

        Raises:
          TypeError: If dtype in spec is not tf.float32 or tf.float64.
        """
        super().__init__(spec, num_values)

        if spec.dtype not in [tf.float32, tf.float64]:
            raise TypeError(
                "FloatSparseTensorGenerator can only generate tf.sparse.SparseTensor with "
                f"dtype in tf.float32 or tf.float64 but found {spec.dtype}."
            )

    def generate(self):
        shape = self._get_shape()
        num_values = self._get_num_values(shape)
        vals = np.random.rand(num_values)
        if self.spec.dtype == tf.float32:
            vals = vals.astype(np.float32)
        coords = self._generate_random_coords(num_values, shape)
        return tf.SparseTensor(indices=coords, values=vals, dense_shape=shape)


class WordSparseTensorGenerator(SparseTensorGeneratorBase):
    """WordSparseTensorGenerator generates string tf.SparseTensor with string
    length similar to a word."""

    def __init__(self, spec, num_values, avg_length=5):
        """Create a new WordSparseTensorGenerator

        WordSparseTensorGenerator samples word length using Poisson distribution
        with lambda equals to avg_length and generates random bytes for each word.

        Args:
          spec: A tf.SparseTensorSpec that describes the output tensor.
          num_values: A value distribution or an int specifying number of non-zero values in the sparse tensor.
          avg_length: An int that represents the average word length.

        Raises:
          TypeError: If dtype in spec is not tf.string.
          ValueError: If avg_length is not positive.
        """
        super().__init__(spec, num_values)

        if spec.dtype is not tf.string:
            raise TypeError(
                "WordSparseTensorGenerator can only generate tf.sparse.SparseTensor with "
                f"dtype in tf.string but found {spec.dtype}."
            )

        if avg_length < 1:
            raise ValueError(
                "WordSparseTensorGenerator must have positive avg_length"
                f" but found {avg_length}."
            )
        self._avg_length = avg_length

    def generate(self):
        # Use Poisson distribution to sample the length of byte strings.
        # The avg_length equals to the lambda in Poisson distribution.
        shape = self._get_shape()
        num_values = self._get_num_values(shape)
        lengths = np.random.poisson(self._avg_length, size=num_values)

        to_string = lambda length: np.random.bytes(length)
        vfunc = np.vectorize(to_string)
        vals = vfunc(lengths)
        coords = self._generate_random_coords(num_values, shape)
        return tf.SparseTensor(indices=coords, values=vals, dense_shape=shape)

    def hash_code(self):
        hash_code = super().hash_code()
        m = hashlib.sha256()
        m.update(hash_code.encode())
        m.update(int_to_bytes(self._avg_length))
        return m.hexdigest()


class BoolSparseTensorGenerator(SparseTensorGeneratorBase):
    """BoolSparseTensorGenerator generates tf.sparse.SparseTensor with dtype in tf.bool."""

    def __init__(self, spec, num_values):
        """Create a new BoolSparseTensorGenerator.

        The generated bool value has equal true and false possibility.

        Args:
          spec: A tf.SparseTensorSpec that describes the output tensor.
          num_values: A value distribution or an int specifying number of non-zero values in the sparse tensor.

        Raises:
          TypeError: If dtype in spec is not tf.bool.
        """
        super().__init__(spec, num_values)

        if spec.dtype is not tf.bool:
            raise TypeError(
                "BoolSparseTensorGenerator can only generate tf.sparse.SparseTensor with "
                f"dtype in tf.bool but found {spec.dtype}."
            )

    def generate(self):
        shape = self._get_shape()
        # np.random.rand generates values from 0 to 1 using Uniform distribution
        num_values = self._get_num_values(shape)
        vals = np.random.rand(num_values) > 0.5
        coords = self._generate_random_coords(num_values, shape)
        return tf.SparseTensor(indices=coords, values=vals, dense_shape=shape)
