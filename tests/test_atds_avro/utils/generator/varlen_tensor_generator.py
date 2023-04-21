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
"""VarLenTensorGenerator"""

import enum
import hashlib
import numpy as np
import tensorflow as tf

from tests.test_atds_avro.utils.generator.generator_base import (
    Generator,
)
from tests.test_atds_avro.utils.hash_util import int_to_bytes


class DimensionDistribution(enum.Enum):
    ONE_DIM = 1  # dimension size 1
    TWO_DIM = 2  # dimension size 2
    LARGE_DIM = 3  # dimension size from 5 to 10


DIM_DISTRIBUTION_TO_RANGE = {DimensionDistribution.LARGE_DIM: (5, 10)}


class VarLenTensorGeneratorBase(Generator):
    """Base of VarLenTensorGeneratorBase that generates tf.sparse.SparseTensor."""

    def __init__(self, spec, dim_dist):
        """Create a new VarLenTensorGeneratorBase.

        This must be called by the constructors of subclasses e.g.
        IntVarLenTensorGeneratorBase, FloatVarLenTensorGeneratorBase, etc.

        Args:
          spec: A tf.SparseTensorSpec that describes the output tensor.
          dim_dist: Distribution of dimension sizes.

        Raises:
          TypeError: If spec is not tf.SparseTensorSpec.
        """
        super().__init__(spec)
        if not isinstance(spec, tf.SparseTensorSpec):
            raise TypeError(
                "Input spec must be a tf.SparseTensorSpec in VarLenTensorGenerator "
                f"but found {spec}"
            )

        if not isinstance(dim_dist, DimensionDistribution):
            raise TypeError(
                f"dim_dist must be a DimensionDistribution but found {dim_dist}"
            )

        if self.spec.shape.rank is None:
            raise ValueError(f"Input spec must have known rank")

        self._dim_dist = dim_dist

    def _get_dim(self):
        if self._dim_dist == DimensionDistribution.ONE_DIM:
            return 1
        elif self._dim_dist == DimensionDistribution.TWO_DIM:
            return 2
        elif self._dim_dist == DimensionDistribution.LARGE_DIM:
            return np.random.randint(*DIM_DISTRIBUTION_TO_RANGE[self._dim_dist])
        else:
            raise ValueError(
                f"Found unsupported dimension distribution {self._dim_dist}"
            )

    def _get_shape(self):
        return [
            dim if dim is not None else self._get_dim()
            for dim in self.spec.shape.as_list()
        ]

    def _get_idx(self, depth, shape, current_idx, ret):
        cur_dim = shape[depth]
        # Generate full list of idx, e.g. for a [2, 3] tensor:
        # [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
        for i in range(cur_dim):
            current_idx[depth] = i
            if depth == len(shape) - 1:
                ret.append(current_idx.copy())
            else:
                self._get_idx(depth + 1, shape, current_idx, ret)

    def hash_code(self):
        hash_code = super().hash_code()

        m = hashlib.sha256()
        m.update(hash_code.encode())
        m.update(DimensionDistribution.__name__.encode())
        m.update(self._dim_dist.name.encode())
        return m.hexdigest()


class IntVarLenTensorGenerator(VarLenTensorGeneratorBase):
    """IntVarLenTensorGenerator generates tf.sparse.SparseTensor with dtype in tf.int32 or tf.int64"""

    def __init__(self, spec, dim_dist=DimensionDistribution.ONE_DIM):
        """Create a new IntVarLenTensorGenerator

        With tf.int32 dtype, the generated int range is between -2^31 to 2^31 - 1.
        With tf.int64 dtype, the generated int range is between -2^63 to 2^63 - 1.

        Args:
          spec: A tf.SparseTensorSpec that describes the output tensor.
          dim_dist: Distribution of dimension sizes.

        Raises:
          TypeError: If dtype in spec is not tf.int32 or tf.int64.
        """
        super().__init__(spec, dim_dist)

        if spec.dtype not in [tf.int32, tf.int64]:
            raise TypeError(
                "IntVarLenTensorGenerator can only generate tf.sparse.SparseTensor with "
                f"dtype in tf.int32 or tf.int64 but found {spec.dtype}."
            )

    def generate(self):
        dtype = self.spec.dtype
        info = np.iinfo(dtype.as_numpy_dtype)
        shape = self._get_shape()
        idxs = []
        self._get_idx(0, shape, [0] * len(shape), idxs)
        vals = np.random.randint(
            low=info.min, high=info.max, size=len(idxs), dtype=dtype.as_numpy_dtype
        )
        return tf.SparseTensor(indices=idxs, values=vals, dense_shape=shape)


class FloatVarLenTensorGenerator(VarLenTensorGeneratorBase):
    """FloatVarLenTensorGenerator generates tf.sparse.SparseTensor with dtype in tf.float32
    or tf.float64."""

    def __init__(self, spec, dim_dist=DimensionDistribution.ONE_DIM):
        """Create a new FloatVarLenTensorGenerator

        The generated float range is between 0.0 to 1.0.

        Args:
          spec: A tf.SparseTensorSpec that describes the output tensor.
          dim_dist: Distribution of dimension sizes.

        Raises:
          TypeError: If dtype in spec is not tf.float32 or tf.float64.
        """
        super().__init__(spec, dim_dist)

        if spec.dtype not in [tf.float32, tf.float64]:
            raise TypeError(
                "FloatVarLenTensorGenerator can only generate tf.sparse.SparseTensor with "
                f"dtype in tf.float32 or tf.float64 but found {spec.dtype}."
            )

    def generate(self):
        shape = self._get_shape()
        idxs = []
        self._get_idx(0, shape, [0] * len(shape), idxs)
        vals = np.random.rand(len(idxs))
        if self.spec.dtype == tf.float32:
            vals = vals.astype(np.float32)
        return tf.SparseTensor(indices=idxs, values=vals, dense_shape=shape)


class WordVarLenTensorGenerator(VarLenTensorGeneratorBase):
    """WordVarLenTensorGenerator generates string tf.SparseTensor with string
    length similar to a word."""

    def __init__(self, spec, dim_dist=DimensionDistribution.ONE_DIM, avg_length=5):
        """Create a new WordVarLenTensorGenerator

        WordVarLenTensorGenerator samples word length using Poisson distribution
        with lambda equals to avg_length and generates random bytes for each word.

        Args:
          spec: A tf.SparseTensorSpec that describes the output tensor.
          dim_dist: Distribution of dimension sizes.
          avg_length: An int that represents the average word length.

        Raises:
          TypeError: If dtype in spec is not tf.string.
          ValueError: If avg_length is not positive.
        """
        super().__init__(spec, dim_dist)

        if spec.dtype is not tf.string:
            raise TypeError(
                "WordVarLenTensorGenerator can only generate tf.sparse.SparseTensor with "
                f"dtype in tf.string but found {spec.dtype}."
            )

        if avg_length < 1:
            raise ValueError(
                "WordVarLenTensorGenerator must have positive avg_length"
                f" but found {avg_length}."
            )
        self._avg_length = avg_length

    def generate(self):
        # Use Poisson distribution to sample the length of byte strings.
        # The avg_length equals to the lambda in Poisson distribution.
        shape = self._get_shape()
        idxs = []
        self._get_idx(0, shape, [0] * len(shape), idxs)
        lengths = np.random.poisson(self._avg_length, size=len(idxs))

        to_string = lambda length: np.random.bytes(length)
        vfunc = np.vectorize(to_string)
        vals = vfunc(lengths)
        return tf.SparseTensor(indices=idxs, values=vals, dense_shape=shape)

    def hash_code(self):
        hash_code = super().hash_code()
        m = hashlib.sha256()
        m.update(hash_code.encode())
        m.update(int_to_bytes(self._avg_length))
        return m.hexdigest()


class BoolVarLenTensorGenerator(VarLenTensorGeneratorBase):
    """BoolVarLenTensorGenerator generates tf.sparse.SparseTensor with dtype in tf.bool."""

    def __init__(self, spec, dim_dist=DimensionDistribution.ONE_DIM):
        """Create a new BoolVarLenTensorGenerator.

    The generated bool value has equal true and false possibility.

    Args:
      spec: A tf.SparseTensorSpec that describes the output tensor.\
      dim_dist: Distribution of dimension sizes.

    Raises:
      TypeError: If dtype in spec is not tf.bool.
    """
        super().__init__(spec, dim_dist)

        if spec.dtype is not tf.bool:
            raise TypeError(
                "BoolVarLenTensorGenerator can only generate tf.sparse.SparseTensor with "
                f"dtype in tf.bool but found {spec.dtype}."
            )

    def generate(self):
        shape = self._get_shape()
        # np.random.rand generates values from 0 to 1 using Uniform distribution
        idxs = []
        self._get_idx(0, shape, [0] * len(shape), idxs)
        vals = np.random.rand(len(idxs)) > 0.5
        return tf.SparseTensor(indices=idxs, values=vals, dense_shape=shape)
