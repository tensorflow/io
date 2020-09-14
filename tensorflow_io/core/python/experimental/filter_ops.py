# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Filter Ops."""

import numpy as np

import tensorflow as tf


def pad(input, ksize, mode, constant_values):
    input = tf.convert_to_tensor(input)
    ksize = tf.convert_to_tensor(ksize)
    mode = "CONSTANT" if mode is None else upper(mode)
    constant_values = (
        tf.zeros([], dtype=input.dtype)
        if constant_values is None
        else tf.convert_to_tensor(constant_values, dtype=input.dtype)
    )

    assert mode in ("CONSTANT", "REFLECT", "SYMMETRIC")

    height, width = ksize[0], ksize[1]
    top = (height - 1) // 2
    bottom = height - 1 - top
    left = (width - 1) // 2
    right = width - 1 - left
    paddings = [[0, 0], [top, bottom], [left, right], [0, 0]]
    return tf.pad(input, paddings, mode=mode, constant_values=constant_values)


def gaussian(input, ksize, sigma, mode=None, constant_values=None, name=None):
    """
    Apply Gaussian filter to image.

    Args:
      input: A 4-D (`[N, H, W, C]`) Tensor.
      ksize: A scalar or 1-D `[kx, ky]` Tensor.
        Size of the Gaussian kernel.
        If scalar, then `ksize` will be broadcasted to 1-D `[kx, ky]`.
      sigma: A scalar or 1-D `[sx, sy]` Tensor.
        Standard deviation for Gaussian kernel.
        If scalar, then `sigma` will be broadcasted to 1-D `[sx, sy]`.
      mode: A `string`. One of "CONSTANT", "REFLECT", or "SYMMETRIC"
        (case-insensitive). Default "CONSTANT".
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode. Must be same type as input. Default 0.
      name: A name for the operation (optional).

    Returns:
      A 4-D (`[N, H, W, C]`) Tensor.
    """

    input = tf.convert_to_tensor(input)
    ksize = tf.convert_to_tensor(ksize)
    sigma = tf.cast(sigma, input.dtype)

    def kernel1d(ksize, sigma, dtype):
        x = tf.range(ksize, dtype=dtype)
        x = x - tf.cast(tf.math.floordiv(ksize, 2), dtype=dtype)
        x = x + tf.where(
            tf.math.equal(tf.math.mod(ksize, 2), 0), tf.cast(0.5, dtype), 0
        )
        g = tf.math.exp(-(tf.math.pow(x, 2) / (2 * tf.math.pow(sigma, 2))))
        g = g / tf.reduce_sum(g)
        return g

    def kernel2d(ksize, sigma, dtype):
        kernel_x = kernel1d(ksize[0], sigma[0], dtype)
        kernel_y = kernel1d(ksize[1], sigma[1], dtype)
        return tf.matmul(
            tf.expand_dims(kernel_x, axis=-1),
            tf.transpose(tf.expand_dims(kernel_y, axis=-1)),
        )

    ksize = tf.broadcast_to(ksize, [2])
    sigma = tf.broadcast_to(sigma, [2])
    g = kernel2d(ksize, sigma, input.dtype)

    input = pad(input, ksize, mode, constant_values)

    channel = tf.shape(input)[-1]
    shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
    g = tf.reshape(g, shape)
    shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
    g = tf.broadcast_to(g, shape)
    return tf.nn.depthwise_conv2d(input, g, [1, 1, 1, 1], padding="VALID")


def laplacian(input, ksize, mode=None, constant_values=None, name=None):
    """
    Apply Laplacian filter to image.

    Args:
      input: A 4-D (`[N, H, W, C]`) Tensor.
      ksize: A scalar Tensor. Kernel size.
      mode: A `string`. One of "CONSTANT", "REFLECT", or "SYMMETRIC"
        (case-insensitive). Default "CONSTANT".
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode. Must be same type as input. Default 0.
      name: A name for the operation (optional).

    Returns:
      A 4-D (`[N, H, W, C]`) Tensor.
    """

    input = tf.convert_to_tensor(input)
    ksize = tf.convert_to_tensor(ksize)

    tf.debugging.assert_none_equal(tf.math.mod(ksize, 2), 0)

    ksize = tf.broadcast_to(ksize, [2])

    total = ksize[0] * ksize[1]
    index = tf.reshape(tf.range(total), ksize)
    g = tf.where(
        tf.math.equal(index, tf.math.floordiv(total - 1, 2)),
        tf.cast(1 - total, input.dtype),
        tf.cast(1, input.dtype),
    )

    input = pad(input, ksize, mode, constant_values)

    channel = tf.shape(input)[-1]
    shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
    g = tf.reshape(g, shape)
    shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
    g = tf.broadcast_to(g, shape)
    return tf.nn.depthwise_conv2d(input, g, [1, 1, 1, 1], padding="VALID")


def gabor(
    input,
    freq,
    sigma=None,
    theta=0,
    nstds=3,
    offset=0,
    mode=None,
    constant_values=None,
    name=None,
):
    """
    Apply Gabor filter to image.

    Args:
      input: A 4-D (`[N, H, W, C]`) Tensor.
      freq: A float Tensor. Spatial frequency of the harmonic function.
        Specified in pixels.
      sigma: A scalar or 1-D `[sx, sy]` Tensor. Standard deviation in
        in x- and y-directions. These directions apply to the kernel
        before rotation. If theta = pi/2, then the kernel is rotated
        90 degrees so that sigma_x controls the vertical direction.
        If scalar, then `sigma` will be broadcasted to 1-D `[sx, sy]`.
      nstd: A scalar Tensor. The linear size of the kernel is nstds
        standard deviations, 3 by default.
      offset: A scalar Tensor. Phase offset of harmonic function in radians.
      mode: A `string`. One of "CONSTANT", "REFLECT", or "SYMMETRIC"
        (case-insensitive). Default "CONSTANT".
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode. Must be same type as input. Default 0.
      name: A name for the operation (optional).

    Returns:
      A 4-D (`[N, H, W, C]`) Tensor.
    """
    input = tf.convert_to_tensor(input)

    dtype = tf.complex128

    freq = tf.cast(freq, dtype.real_dtype)
    if sigma is None:
        # See http://www.cs.rug.nl/~imaging/simplecell.html
        b = 1  # bandwidth
        sigma = (
            tf.cast(
                1.0
                / np.pi
                * np.sqrt(np.log(2) / 2.0)
                * (2.0 ** b + 1)
                / (2.0 ** b - 1),
                dtype.real_dtype,
            )
            / freq
        )
    sigma = tf.broadcast_to(sigma, [2])
    sigma_x, sigma_y = sigma[0], sigma[1]
    theta = tf.cast(theta, dtype.real_dtype)
    nstds = tf.cast(nstds, dtype.real_dtype)
    offset = tf.cast(offset, dtype.real_dtype)

    x0 = tf.math.ceil(
        tf.math.maximum(
            tf.math.abs(nstds * sigma_x * tf.math.cos(theta)),
            tf.math.abs(nstds * sigma_y * tf.math.sin(theta)),
            tf.cast(1, dtype.real_dtype),
        )
    )
    y0 = tf.math.ceil(
        tf.math.maximum(
            tf.math.abs(nstds * sigma_y * tf.math.cos(theta)),
            tf.math.abs(nstds * sigma_x * tf.math.sin(theta)),
            tf.cast(1, dtype.real_dtype),
        )
    )
    y, x = tf.meshgrid(tf.range(-y0, y0 + 1), tf.range(-x0, x0 + 1))
    y, x = tf.transpose(y), tf.transpose(x)

    rotx = y * tf.math.sin(theta) + x * tf.math.cos(theta)
    roty = y * tf.math.cos(theta) - x * tf.math.sin(theta)

    g = tf.math.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    g = g / (2 * np.pi * sigma_x * sigma_y)
    g = tf.cast(g, dtype) * tf.exp(
        tf.cast(1j, dtype) * tf.cast(2 * np.pi * freq * rotx + offset, dtype)
    )

    ksize = tf.shape(g)

    input = pad(input, ksize, mode, constant_values)

    channel = tf.shape(input)[-1]
    shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
    g = tf.reshape(g, shape)
    shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
    g = tf.broadcast_to(g, shape)

    real = tf.nn.depthwise_conv2d(
        input, tf.cast(tf.math.real(g), input.dtype), [1, 1, 1, 1], padding="VALID"
    )
    imag = tf.nn.depthwise_conv2d(
        input, tf.cast(tf.math.imag(g), input.dtype), [1, 1, 1, 1], padding="VALID"
    )
    return tf.complex(real, imag)


def prewitt(input, mode=None, constant_values=None, name=None):
    """
    Apply Prewitt filter to image.

    Args:
      input: A 4-D (`[N, H, W, C]`) Tensor.
      mode: A `string`. One of "CONSTANT", "REFLECT", or "SYMMETRIC"
        (case-insensitive). Default "CONSTANT".
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode. Must be same type as input. Default 0.
      name: A name for the operation (optional).

    Returns:
      A 4-D (`[N, H, W, C]`) Tensor.
    """

    input = tf.convert_to_tensor(input)

    gx = tf.cast([[1, 0, -1], [1, 0, -1], [1, 0, -1]], input.dtype)
    gy = tf.cast([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], input.dtype)

    ksize = tf.constant([3, 3])

    input = pad(input, ksize, mode, constant_values)

    channel = tf.shape(input)[-1]
    shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
    gx, gy = tf.reshape(gx, shape), tf.reshape(gy, shape)
    shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
    gx, gy = tf.broadcast_to(gx, shape), tf.broadcast_to(gy, shape)

    x = tf.nn.depthwise_conv2d(
        input, tf.cast(gx, input.dtype), [1, 1, 1, 1], padding="VALID"
    )
    y = tf.nn.depthwise_conv2d(
        input, tf.cast(gy, input.dtype), [1, 1, 1, 1], padding="VALID"
    )
    return tf.math.sqrt(x * x + y * y)


def sobel(input, mode=None, constant_values=None, name=None):
    """
    Apply Sobel filter to image.

    Args:
      input: A 4-D (`[N, H, W, C]`) Tensor.
      mode: A `string`. One of "CONSTANT", "REFLECT", or "SYMMETRIC"
        (case-insensitive). Default "CONSTANT".
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode. Must be same type as input. Default 0.
      name: A name for the operation (optional).

    Returns:
      A 4-D (`[N, H, W, C]`) Tensor.
    """

    input = tf.convert_to_tensor(input)

    gx = tf.cast([[1, 0, -1], [2, 0, -2], [1, 0, -1]], input.dtype)
    gy = tf.cast([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], input.dtype)

    ksize = tf.constant([3, 3])

    input = pad(input, ksize, mode, constant_values)

    channel = tf.shape(input)[-1]
    shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
    gx, gy = tf.reshape(gx, shape), tf.reshape(gy, shape)
    shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
    gx, gy = tf.broadcast_to(gx, shape), tf.broadcast_to(gy, shape)

    x = tf.nn.depthwise_conv2d(
        input, tf.cast(gx, input.dtype), [1, 1, 1, 1], padding="VALID"
    )
    y = tf.nn.depthwise_conv2d(
        input, tf.cast(gy, input.dtype), [1, 1, 1, 1], padding="VALID"
    )
    return tf.math.sqrt(x * x + y * y)
