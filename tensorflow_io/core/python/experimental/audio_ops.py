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

import numpy as np

import tensorflow as tf


def spectrogram(input, nfft, window, stride, name=None):
    """
    Create spectrogram from audio.

    Args:
      input: An 1-D audio signal Tensor.
      nfft: Size of FFT.
      window: Size of window.
      stride: Size of hops between windows.
      name: A name for the operation (optional).

    Returns:
      A tensor of spectrogram.
    """

    # TODO: Support audio with channel > 1.
    return tf.math.abs(
        tf.signal.stft(
            input,
            frame_length=window,
            frame_step=stride,
            fft_length=nfft,
            window_fn=tf.signal.hann_window,
            pad_end=True,
        )
    )


def melscale(input, rate, mels, fmin, fmax, name=None):
    """
    Turn spectrogram into mel scale spectrogram

    Args:
      input: A spectrogram Tensor with shape [frames, nfft+1].
      rate: Sample rate of the audio.
      mels: Number of mel filterbanks.
      fmin: Minimum frequency. 
      fmax: Maximum frequency. 
      name: A name for the operation (optional).

    Returns:
      A tensor of mel spectrogram with shape [frames, mels].
    """

    # TODO: Support audio with channel > 1.
    nbins = tf.shape(input)[-1]
    matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mels,
        num_spectrogram_bins=nbins,
        sample_rate=rate,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
    )

    return tf.tensordot(input, matrix, 1)


def dbscale(input, top_db, name=None):
    """
    Turn spectrogram into db scale

    Args:
      input: A spectrogram Tensor.
      top_db: Minimum negative cut-off `max(10 * log10(S)) - top_db`
      name: A name for the operation (optional).

    Returns:
      A tensor of mel spectrogram with shape [frames, mels].
    """
    power = tf.math.square(input)
    log_spec = 10.0 * (tf.math.log(power) - tf.math.log(10.0))
    log_spec = tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - top_db)
    return log_spec


def freq_mask(input, param, name=None):
    """
    Apply masking to a spectrogram in the freq domain.

    Args:
      input: An audio spectogram.
      param: Parameter of freq masking.
      name: A name for the operation (optional).
    Returns:
      A tensor of spectrogram.
    """
    # TODO: Support audio with channel > 1.
    freq_max = tf.shape(input)[1]
    f = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
    f0 = tf.random.uniform(
        shape=(), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32
    )
    indices = tf.reshape(tf.range(freq_max), (1, -1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
    )
    return tf.where(condition, 0, input)


def time_mask(input, param, name=None):
    """
    Apply masking to a spectrogram in the time domain.

    Args:
      input: An audio spectogram.
      param: Parameter of time masking.
      name: A name for the operation (optional).
    Returns:
      A tensor of spectrogram.
    """
    # TODO: Support audio with channel > 1.
    time_max = tf.shape(input)[0]
    t = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
    t0 = tf.random.uniform(
        shape=(), minval=0, maxval=time_max - t, dtype=tf.dtypes.int32
    )
    indices = tf.reshape(tf.range(time_max), (-1, 1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, t0), tf.math.less(indices, t0 + t)
    )


def fade(input, fade_in, fade_out, mode, name=None):
    """
    Apply fade in/out to audio.

    Args:
      input: An audio spectogram.
      fade_in: Length of fade in.
      fade_out: Length of fade out.
      mode: Mode of the fade.
      name: A name for the operation (optional).
    Returns:
      A tensor of audio.
    """
    # TODO length may not be at axis=0, if `batch` (axis=0) is present.
    axis = 0

    shape = tf.shape(input)
    length = shape[axis]

    ones_in = tf.ones([length - fade_in])
    factor_in = tf.linspace(0.0, 1.0, fade_in)
    if mode == "linear":
        factor_in = factor_in
    elif mode == "logarithmic":
        factor_in = tf.math.log1p(factor_in) / tf.math.log1p(1.0)
    elif mode == "exponential":
        factor_in = tf.math.expm1(factor_in) / tf.math.expm1(1.0)
    else:
        raise ValueError("{} mode not supported".format(mode))

    factor_in = tf.concat([factor_in, ones_in], axis=0)

    ones_out = tf.ones([length - fade_out])
    factor_out = 1.0 - tf.linspace(0.0, 1.0, fade_out)
    if mode == "linear":
        factor_out = factor_out
    elif mode == "logarithmic":
        factor_out = tf.math.log1p(factor_out) / tf.math.log1p(1.0)
    elif mode == "exponential":
        factor_out = tf.math.expm1(factor_out) / tf.math.expm1(1.0)
    else:
        raise ValueError("{} mode not supported".format(mode))

    factor_out = tf.concat([ones_out, factor_out], axis=0)

    # reshape to get to the same rank, then broadcast to shape
    rank = tf.cast(tf.rank(input), tf.int64)
    factor_in = tf.reshape(
        factor_in, tf.where(tf.math.equal(tf.range(rank), axis), shape, 1)
    )
    factor_in = tf.broadcast_to(factor_in, shape)
    factor_out = tf.reshape(
        factor_out, tf.where(tf.math.equal(tf.range(rank), axis), shape, 1)
    )
    factor_out = tf.broadcast_to(factor_out, shape)

    return factor_in * factor_out * input
