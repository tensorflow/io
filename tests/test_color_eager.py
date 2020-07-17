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
"""Test tfio.experimental.color"""

import os
import shutil
import tempfile
import numpy as np
import pytest

import skimage.color

import tensorflow as tf
import tensorflow_io as tfio


@pytest.mark.parametrize(
    ("data", "func", "check"),
    [
        pytest.param(
            lambda: (np.random.random((10, 20, 3)) * 256.0).astype(np.uint8),
            tfio.experimental.color.rgb_to_bgr,
            lambda e: e[..., ::-1],
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3)) * 256.0).astype(np.uint8),
            tfio.experimental.color.bgr_to_rgb,
            lambda e: e[..., ::-1],
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3)) * 256.0).astype(np.uint8),
            tfio.experimental.color.rgb_to_rgba,
            lambda e: np.dstack([e, np.zeros(shape=[10, 20])]),
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 4)) * 256.0).astype(np.uint8),
            tfio.experimental.color.rgba_to_rgb,
            lambda e: e[..., 0:3],
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3)) * 256.0).astype(np.uint8),
            tfio.experimental.color.rgb_to_ycbcr,
            lambda e: skimage.color.rgb2ycbcr(e).astype(np.uint8),
        ),
        pytest.param(
            lambda: np.array([[[117, 147, 67]]], np.uint8),
            tfio.experimental.color.ycbcr_to_rgb,
            lambda e: (skimage.color.ycbcr2rgb(e.astype(np.float32)) * 255.0).astype(
                np.uint8
            ),
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.rgb_to_ypbpr,
            skimage.color.rgb2ypbpr,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.ypbpr_to_rgb,
            skimage.color.ypbpr2rgb,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.rgb_to_ydbdr,
            skimage.color.rgb2ydbdr,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.ydbdr_to_rgb,
            skimage.color.ydbdr2rgb,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.rgb_to_hsv,
            skimage.color.rgb2hsv,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.hsv_to_rgb,
            skimage.color.hsv2rgb,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.rgb_to_yiq,
            skimage.color.rgb2yiq,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.yiq_to_rgb,
            skimage.color.yiq2rgb,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.rgb_to_yuv,
            skimage.color.rgb2yuv,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.yuv_to_rgb,
            skimage.color.yuv2rgb,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.rgb_to_xyz,
            skimage.color.rgb2xyz,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.xyz_to_rgb,
            skimage.color.xyz2rgb,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.rgb_to_lab,
            skimage.color.rgb2lab,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.lab_to_rgb,
            skimage.color.lab2rgb,
        ),
        pytest.param(
            lambda: (np.random.random((10, 20, 3))).astype(np.float32),
            tfio.experimental.color.rgb_to_grayscale,
            lambda e: tf.expand_dims(skimage.color.rgb2gray(e), axis=-1),
        ),
    ],
    ids=[
        "rgb_to_bgr",
        "bgr_to_rgb",
        "rgb_to_rgba",
        "rgba_to_rgb",
        "rgb_to_ycbcr",
        "ycbcr_to_rgb",
        "rgb_to_ypbpr",
        "ypbpr_to_rgb",
        "rgb_to_ydbdr",
        "ydbdr_to_rgb",
        "rgb_to_hsv",
        "hsv_to_rgb",
        "rgb_to_yiq",
        "yiq_to_rgb",
        "rgb_to_yuv",
        "yuv_to_rgb",
        "rgb_to_xyz",
        "xyz_to_rgb",
        "rgb_to_lab",
        "lab_to_rgb",
        "rgb_to_grayscale",
    ],
)
def test_color(data, func, check):
    """test_io_color"""

    np.random.seed(1000)

    input_3d = data()
    expected_3d = check(input_3d)

    output_3d = func(input_3d)
    if input_3d.dtype == np.float32:
        assert np.allclose(output_3d, expected_3d, rtol=0.03)
    else:
        assert np.array_equal(output_3d, expected_3d)

    input_4d = tf.expand_dims(input_3d, axis=0)
    expected_4d = tf.expand_dims(expected_3d, axis=0)

    output_4d = func(input_4d)
    if input_4d.dtype == np.float32:
        assert np.allclose(output_4d, expected_4d, rtol=0.03)
    else:
        assert np.array_equal(output_4d, expected_4d)
