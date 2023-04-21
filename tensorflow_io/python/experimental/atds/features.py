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
import collections
from typing import List

import tensorflow as tf

ATDS_SUPPORTED_DTYPES = [tf.int32, tf.int64, tf.float32, tf.float64, tf.string, tf.bool]


def _raise_error_if_dtype_not_supported(dtype: tf.dtypes.DType):
    if dtype not in ATDS_SUPPORTED_DTYPES:
        raise ValueError(
            f"{dtype} is not supported in ATDS. "
            f"Available dtypes are {ATDS_SUPPORTED_DTYPES}."
        )


def _raise_error_if_shape_is_none(shape: List[int]):
    if shape is None:
        raise ValueError(f"Shape cannot be None.")


def _raise_error_if_shape_has_none_or_zero(shape: List[int]):
    for dim in shape:
        if dim is None or dim == 0:
            raise ValueError(
                f"Dimension in shape cannot be None or 0 but found {shape}."
            )


def _raise_error_if_dtype_is_none(dtype: tf.dtypes.DType):
    if dtype is None:
        raise ValueError(f"dtype cannot be None.")


def _validate_shape_and_dtype(shape: List[int], dtype: tf.dtypes.DType):
    _raise_error_if_shape_is_none(shape)
    _raise_error_if_shape_has_none_or_zero(shape)
    _raise_error_if_dtype_is_none(dtype)
    _raise_error_if_dtype_not_supported(dtype)


class DenseFeature(collections.namedtuple("DenseFeature", ["shape", "dtype"])):
    """
    Configuration for reading and parsing a tf.Tensor encoded with
    ATDS dense feature schema.

    Fields:
      shape: Shape of input data. Each dimension should be positive.
      dtype: Data type of input.
    """

    def __new__(cls, shape: List[int], dtype: tf.dtypes.DType):
        _validate_shape_and_dtype(shape, dtype)
        for dim in shape:
            if dim <= 0:
                raise ValueError(
                    f"Each dimension should be greater than 0"
                    f" in DenseFeature but found {shape}."
                )

        return super().__new__(cls, shape, dtype)


class SparseFeature(collections.namedtuple("SparseFeature", ["shape", "dtype"])):
    """
    Configuration for reading and parsing a tf.SparseTensor encoded with
    ATDS sparse feature schema.

    Fields:
      shape: Shape of input data. shape cannot be empty.
      dtype: Data type of input.
    """

    def __new__(cls, shape: List[int], dtype: tf.dtypes.DType):
        _validate_shape_and_dtype(shape, dtype)
        if len(shape) == 0:
            raise ValueError("SparseFeature cannot be scalar.")

        return super().__new__(cls, shape, dtype)


class VarlenFeature(collections.namedtuple("VarlenFeature", ["shape", "dtype"])):
    """
    Configuration for reading and parsing a tf.SparseTensor encoded with
    ATDS ragged feature schema.

    Fields:
      shape: Shape of input data. Use -1 as unknown dimension.
      dtype: Data type of input.
    """

    def __new__(cls, shape: List[int], dtype: tf.dtypes.DType):
        _validate_shape_and_dtype(shape, dtype)
        for dim in shape:
            if dim <= 0 and dim != -1:
                raise ValueError(
                    f"Each dimension should be greater than 0 or "
                    f"-1 in VarlenFeature but found {shape}."
                )

        return super().__new__(cls, shape, dtype)
