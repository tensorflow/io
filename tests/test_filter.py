# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test tfio.experimental.filter"""

import os
import shutil
import tempfile
import numpy as np
import pytest

import scipy
import skimage.filters

import tensorflow as tf
import tensorflow_io as tfio


@pytest.mark.parametrize(
    ("data", "func", "check"),
    [
        pytest.param(
            lambda: (np.reshape(np.arange(75) / 75, [1, 5, 5, 3])),
            lambda e: tfio.experimental.filter.gaussian(e, ksize=(5, 5), sigma=1),
            lambda e: skimage.filters.gaussian(
                np.reshape(e, [5, 5, 3]),
                sigma=1,
                preserve_range=False,
                channel_axis=-1,
                mode="constant",
            ),
        ),
        pytest.param(
            lambda: (np.reshape(np.arange(25) / 25, [1, 5, 5, 1])),
            lambda e: tfio.experimental.filter.laplacian(e, ksize=(3, 3)),
            lambda e: np.reshape(
                scipy.ndimage.convolve(
                    np.reshape(e, [5, 5]),
                    [[1, 1, 1], [1, -8, 1], [1, 1, 1]],
                    mode="constant",
                ),
                [1, 5, 5, 1],
            ),
        ),
        pytest.param(
            lambda: (np.reshape(np.arange(25) / 25, [1, 5, 5, 1])),
            lambda e: tf.math.real(tfio.experimental.filter.gabor(e, freq=1)),
            lambda e: np.reshape(
                skimage.filters.gabor(
                    np.reshape(e, [5, 5]), frequency=1, mode="constant"
                )[0],
                [1, 5, 5, 1],
            ),
        ),
        pytest.param(
            lambda: (np.reshape(np.arange(25) / 25, [1, 5, 5, 1])),
            lambda e: tfio.experimental.filter.prewitt(e),
            lambda e: np.reshape(
                np.sqrt(
                    np.square(
                        scipy.ndimage.prewitt(
                            np.reshape(e, [5, 5]), axis=0, mode="constant"
                        )
                    )
                    + np.square(
                        scipy.ndimage.prewitt(
                            np.reshape(e, [5, 5]), axis=1, mode="constant"
                        )
                    )
                ),
                [1, 5, 5, 1],
            ),
        ),
        pytest.param(
            lambda: (np.reshape(np.arange(25) / 25, [1, 5, 5, 1])),
            lambda e: tfio.experimental.filter.sobel(e),
            lambda e: np.reshape(
                np.sqrt(
                    np.square(
                        scipy.ndimage.sobel(
                            np.reshape(e, [5, 5]), axis=0, mode="constant"
                        )
                    )
                    + np.square(
                        scipy.ndimage.sobel(
                            np.reshape(e, [5, 5]), axis=1, mode="constant"
                        )
                    )
                ),
                [1, 5, 5, 1],
            ),
        ),
    ],
    ids=["gaussian", "laplacian", "gabor", "prewitt", "sobel"],
)
def test_filter(data, func, check):
    """test_filter"""

    np.random.seed(1000)

    input_4d = data()
    expected_4d = check(input_4d)

    output_4d = func(input_4d)
    assert np.allclose(output_4d, expected_4d, atol=0.02)
