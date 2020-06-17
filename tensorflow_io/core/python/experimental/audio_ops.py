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
"""Audio Ops."""

import tensorflow as tf


def decode_mulaw(input, quantization=None, name=None):
    """
    Decode mu law encoded single (ITU-T, 1988).

    Args:
      input: A signal to be decoded.
      quantization: Number of channels. Default 256.
      name: A name for the operation (optional).

    Returns:
      A mu law decoded signal.
    """
    if quantization is None:
        quantization = 256
    mu = tf.cast(quantization, tf.float32) - 1
    y = (tf.cast(input, tf.float32) / mu) * 2 - 1.0
    x = tf.sign(y) * (tf.math.expm1(tf.math.abs(y) * tf.math.log1p(mu))) / mu
    return x


def encode_mulaw(input, quantization=None, name=None):
    """
    Perform mu law companding transformation (ITU-T, 1988).

    Args:
      input: A signal to be encoded.
      quantization: Number of channels. Default 256.
      name: A name for the operation (optional).

    Returns:
      A mu law encoded signal.
    """
    if quantization is None:
        quantization = 256
    mu = tf.cast(quantization, tf.float32) - 1
    x = tf.cast(input, tf.float32)
    y = tf.sign(x) * tf.math.log1p(mu * tf.math.abs(x)) / tf.math.log1p(mu)
    return tf.cast((y + 1) / 2 * mu + 0.5, tf.int32)
