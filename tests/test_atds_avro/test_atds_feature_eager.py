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
import pytest
import re
import tensorflow as tf

from tensorflow_io.python.experimental.atds.features import (
    DenseFeature,
    SparseFeature,
    VarlenFeature,
    ATDS_SUPPORTED_DTYPES,
)


@pytest.mark.parametrize(
    ["shape", "dtype", "error_message"],
    [
        (None, tf.int32, "Shape cannot be None."),
        ([1, 2], None, "dtype cannot be None."),
        (
            [1, None],
            tf.int32,
            "Dimension in shape cannot be None or 0 but found [1, None].",
        ),
        ([1, 0], tf.int32, "Dimension in shape cannot be None or 0 but found [1, 0]."),
        (
            [1, 2],
            tf.float16,
            f"<dtype: 'float16'> is not supported in ATDS. Available dtypes are {ATDS_SUPPORTED_DTYPES}.",
        ),
        (
            [-1, 2],
            tf.int32,
            "Each dimension should be greater than 0 in DenseFeature but found [-1, 2].",
        ),
    ],
)
def test_atds_dense_feature(shape, dtype, error_message):
    """test DenseFeature creation"""
    with pytest.raises(ValueError, match=re.escape(error_message)):
        DenseFeature(shape, dtype)


@pytest.mark.parametrize(
    ["shape", "dtype", "error_message"],
    [
        (None, tf.int32, "Shape cannot be None."),
        ([1, 2], None, "dtype cannot be None."),
        (
            [1, None],
            tf.int32,
            "Dimension in shape cannot be None or 0 but found [1, None].",
        ),
        ([3, 0], tf.int32, "Dimension in shape cannot be None or 0 but found [3, 0]."),
        (
            [1, 2],
            tf.float16,
            f"<dtype: 'float16'> is not supported in ATDS. Available dtypes are {ATDS_SUPPORTED_DTYPES}.",
        ),
        ([], tf.int64, "SparseFeature cannot be scalar."),
    ],
)
def test_atds_sparse_feature(shape, dtype, error_message):
    """test SparseFeature creation"""
    with pytest.raises(ValueError, match=re.escape(error_message)):
        SparseFeature(shape, dtype)


@pytest.mark.parametrize(
    ["shape", "dtype", "error_message"],
    [
        (None, tf.int32, "Shape cannot be None."),
        ([1, 2], None, "dtype cannot be None."),
        (
            [1, None],
            tf.int32,
            "Dimension in shape cannot be None or 0 but found [1, None].",
        ),
        ([0, 1], tf.int32, "Dimension in shape cannot be None or 0 but found [0, 1]."),
        (
            [-1, 2],
            tf.float16,
            f"<dtype: 'float16'> is not supported in ATDS. Available dtypes are {ATDS_SUPPORTED_DTYPES}.",
        ),
        (
            [-2, 5],
            tf.int32,
            "Each dimension should be greater than 0 or -1 in VarlenFeature but found [-2, 5].",
        ),
    ],
)
def test_atds_ragged_feature(shape, dtype, error_message):
    """test VarlenFeature creation"""
    with pytest.raises(ValueError, match=re.escape(error_message)):
        VarlenFeature(shape, dtype)
