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
"""Color Space Ops."""

import tensorflow as tf


def rgb_to_bgr(input, name=None):
    """
    Convert a RGB image to BGR.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    rgb = tf.unstack(input, axis=-1)
    r, g, b = rgb[0], rgb[1], rgb[2]
    return tf.stack([b, g, r], axis=-1)


def bgr_to_rgb(input, name=None):
    """
    Convert a BGR image to RGB.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    bgr = tf.unstack(input, axis=-1)
    b, g, r = bgr[0], bgr[1], bgr[2]
    return tf.stack([r, g, b], axis=-1)


def rgb_to_rgba(input, name=None):
    """
    Convert a RGB image to RGBA.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 4]`) or 4-D (`[N, H, W, 4]`) Tensor.
    """
    rgb = tf.unstack(input, axis=-1)
    r, g, b = rgb[0], rgb[1], rgb[2]
    a = tf.zeros_like(r)
    return tf.stack([r, g, b, a], axis=-1)


def rgba_to_rgb(input, name=None):
    """
    Convert a RGBA image to RGB.

    Args:
      input: A 3-D (`[H, W, 4]`) or 4-D (`[N, H, W, 4]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    rgba = tf.unstack(input, axis=-1)
    r, g, b, a = rgba[0], rgba[1], rgba[2], rgba[3]
    return tf.stack([r, g, b], axis=-1)


def rgb_to_ycbcr(input, name=None):
    """
    Convert a RGB image to YCbCr.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)

    assert input.dtype == tf.uint8
    value = tf.cast(input, tf.float32)
    value = value / 255.0
    value = rgb_to_ypbpr(value)
    value = value * tf.constant([219, 224, 224], value.dtype)
    value = value + tf.constant([16, 128, 128], value.dtype)
    return tf.cast(value, input.dtype)


def ycbcr_to_rgb(input, name=None):
    """
    Convert a YCbCr image to RGB.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)

    assert input.dtype == tf.uint8
    value = tf.cast(input, tf.float32)
    value = value - tf.constant([16, 128, 128], value.dtype)
    value = value / tf.constant([219, 224, 224], value.dtype)
    value = ypbpr_to_rgb(value)
    value = value * 255.0
    return tf.cast(value, input.dtype)


def rgb_to_ypbpr(input, name=None):
    """
    Convert a RGB image to YPbPr.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)

    kernel = tf.constant(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ],
        input.dtype,
    )

    return tf.tensordot(input, tf.transpose(kernel), axes=((-1,), (0,)))


def ypbpr_to_rgb(input, name=None):
    """
    Convert a YPbPr image to RGB.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)

    # inv of:
    # [[ 0.299   , 0.587   , 0.114   ],
    #  [-0.168736,-0.331264, 0.5     ],
    #  [ 0.5     ,-0.418688,-0.081312]]
    kernel = tf.constant(
        [
            [1.00000000e00, -1.21889419e-06, 1.40199959e00],
            [1.00000000e00, -3.44135678e-01, -7.14136156e-01],
            [1.00000000e00, 1.77200007e00, 4.06298063e-07],
        ],
        input.dtype,
    )

    return tf.tensordot(input, tf.transpose(kernel), axes=((-1,), (0,)))


def rgb_to_ydbdr(input, name=None):
    """
    Convert a RGB image to YDbDr.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)

    kernel = tf.constant(
        [[0.299, 0.587, 0.114], [-0.45, -0.883, 1.333], [-1.333, 1.116, 0.217]],
        input.dtype,
    )

    return tf.tensordot(input, tf.transpose(kernel), axes=((-1,), (0,)))


def ydbdr_to_rgb(input, name=None):
    """
    Convert a YDbDr image to RGB.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)

    # inv of:
    # [[    0.299,   0.587,    0.114],
    #  [   -0.45 ,  -0.883,    1.333],
    #  [   -1.333,   1.116,    0.217]]
    kernel = tf.constant(
        [
            [1.00000000e00, 9.23037161e-05, -5.25912631e-01],
            [1.00000000e00, -1.29132899e-01, 2.67899328e-01],
            [1.00000000e00, 6.64679060e-01, -7.92025435e-05],
        ],
        input.dtype,
    )

    return tf.tensordot(input, tf.transpose(kernel), axes=((-1,), (0,)))


def rgb_to_hsv(input, name=None):
    """
    Convert a RGB image to HSV.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: Alias to tf.image.rgb_to_hsv for completeness
    input = tf.convert_to_tensor(input)
    return tf.image.rgb_to_hsv(input)


def hsv_to_rgb(input, name=None):
    """
    Convert a HSV image to RGB.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: Alias to tf.image.hsv_to_rgb for completeness
    input = tf.convert_to_tensor(input)
    return tf.image.hsv_to_rgb(input)


def rgb_to_yiq(input, name=None):
    """
    Convert a RGB image to YIQ.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: Alias to tf.image.rgb_to_yiq for completeness
    input = tf.convert_to_tensor(input)
    return tf.image.rgb_to_yiq(input)


def yiq_to_rgb(input, name=None):
    """
    Convert a YIQ image to RGB.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: Alias to tf.image.yiq_to_rgb for completeness
    input = tf.convert_to_tensor(input)
    return tf.image.yiq_to_rgb(input)


def rgb_to_yuv(input, name=None):
    """
    Convert a RGB image to YUV.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: Alias to tf.image.rgb_to_yuv for completeness
    input = tf.convert_to_tensor(input)
    return tf.image.rgb_to_yuv(input)


def yuv_to_rgb(input, name=None):
    """
    Convert a YUV image to RGB.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: Alias to tf.image.yuv_to_rgb for completeness
    input = tf.convert_to_tensor(input)
    return tf.image.yuv_to_rgb(input)


def rgb_to_xyz(input, name=None):
    """
    Convert a RGB image to CIE XYZ.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)

    kernel = tf.constant(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ],
        input.dtype,
    )
    value = tf.where(
        tf.math.greater(input, 0.04045),
        tf.math.pow((input + 0.055) / 1.055, 2.4),
        input / 12.92,
    )
    return tf.tensordot(value, tf.transpose(kernel), axes=((-1,), (0,)))


def xyz_to_rgb(input, name=None):
    """
    Convert a CIE XYZ image to RGB.

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)

    # inv of:
    # [[0.412453, 0.35758 , 0.180423],
    #  [0.212671, 0.71516 , 0.072169],
    #  [0.019334, 0.119193, 0.950227]]
    kernel = tf.constant(
        [
            [3.24048134, -1.53715152, -0.49853633],
            [-0.96925495, 1.87599, 0.04155593],
            [0.05564664, -0.20404134, 1.05731107],
        ],
        input.dtype,
    )
    value = tf.tensordot(input, tf.transpose(kernel), axes=((-1,), (0,)))
    value = tf.where(
        tf.math.greater(value, 0.0031308),
        tf.math.pow(value, 1.0 / 2.4) * 1.055 - 0.055,
        value * 12.92,
    )
    return tf.clip_by_value(value, 0, 1)


def rgb_to_grayscale(input, name=None):
    """
    Convert a RGB image to Grayscale (ITU-R).

    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    # Note: This rgb_to_grayscale conforms to skimage.color.rgb2gray
    # and is different from tf.image.rgb_to_grayscale
    input = tf.convert_to_tensor(input)

    value = tf.image.convert_image_dtype(input, tf.float32)
    coeff = [0.2125, 0.7154, 0.0721]
    value = tf.tensordot(value, coeff, (-1, -1))
    value = tf.expand_dims(value, -1)
    return tf.image.convert_image_dtype(value, input.dtype)
