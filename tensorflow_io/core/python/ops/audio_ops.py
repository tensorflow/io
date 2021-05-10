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
"""audio"""

import sys

import tensorflow as tf

from tensorflow_io.core.python.ops import core_ops


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
    log_spec = 10.0 * (tf.math.log(power) / tf.math.log(10.0))
    log_spec = tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - top_db)
    return log_spec


def remix(input, axis, indices, name=None):
    """
    Remix the audio from segments indices.

    Args:
      input: An audio Tensor.
      axis: The axis to trim.
      indices: The indices of `start, stop` of each segments.
      name: A name for the operation (optional).
    Returns:
      A tensor of remixed audio.
    """
    shape = tf.shape(indices, out_type=tf.int64)

    rank = tf.cast(tf.rank(indices), tf.int64)
    mask = tf.math.equal(tf.range(rank), axis + 1)

    start = tf.slice(
        indices,
        tf.where(mask, tf.cast(0, tf.int64), 0),
        tf.where(mask, tf.cast(1, tf.int64), shape),
    )
    stop = tf.slice(
        indices,
        tf.where(mask, tf.cast(1, tf.int64), 0),
        tf.where(mask, tf.cast(1, tf.int64), shape),
    )

    start = tf.squeeze(start, axis=[axis + 1])
    stop = tf.squeeze(stop, axis=[axis + 1])

    start = tf.expand_dims(start, axis=axis)
    stop = tf.expand_dims(stop, axis=axis)

    shape = tf.shape(input, out_type=tf.int64)
    length = shape[axis]

    rank = tf.cast(tf.rank(input), tf.int64)
    indices = tf.range(length, dtype=tf.int64)
    indices = tf.reshape(
        indices, tf.where(tf.math.equal(tf.range(rank), axis), length, 1)
    )
    indices = tf.broadcast_to(indices, shape)

    indices = tf.expand_dims(indices, axis=axis + 1)

    mask = tf.math.logical_and(
        tf.math.greater_equal(indices, start), tf.math.less(indices, stop)
    )

    mask = tf.reduce_any(mask, axis=axis + 1)

    # count bool to adjust padding
    count = tf.reduce_sum(tf.cast(mask, tf.int64), axis=axis, keepdims=True)

    # length after padding
    length = tf.reduce_max(count)

    # delta
    delta = count - tf.reduce_min(count)
    padding = tf.range(tf.constant(1, tf.int64), tf.reduce_max(delta) + 1)
    padding = tf.reshape(
        padding, tf.where(tf.math.equal(tf.range(rank), axis), tf.reduce_max(delta), 1)
    )
    padding = tf.broadcast_to(
        padding,
        tf.where(tf.math.equal(tf.range(rank), axis), tf.reduce_max(delta), shape),
    )
    padding = tf.math.greater(padding, delta)

    mask = tf.concat([mask, padding], axis=axis)
    input = tf.concat([input, tf.zeros(tf.shape(padding), input.dtype)], axis=axis)
    result = tf.boolean_mask(input, mask)
    result = tf.reshape(
        result, tf.where(tf.math.equal(tf.range(rank), axis), length, shape)
    )

    return result


def split(input, axis, epsilon, name=None):
    """
    Split the audio by removing the noise smaller than epsilon.

    Args:
      input: An audio Tensor.
      axis: The axis to trim.
      epsilon: The max value to be considered as noise.
      name: A name for the operation (optional).
    Returns:
      A tensor of start and stop with shape `[..., 2, ...]`.
    """
    shape = tf.shape(input, out_type=tf.int64)
    length = shape[axis]

    nonzero = tf.math.greater(input, epsilon)

    rank = tf.cast(tf.rank(input), tf.int64)
    mask = tf.math.equal(tf.range(rank), axis)

    fill = tf.zeros(tf.where(mask, 1, shape), tf.int8)

    curr = tf.cast(nonzero, tf.int8)
    prev = tf.concat(
        [
            fill,
            tf.slice(
                curr,
                tf.where(mask, tf.constant(0, tf.int64), 0),
                tf.where(mask, length - 1, shape),
            ),
        ],
        axis=axis,
    )
    next = tf.concat(
        [
            tf.slice(
                curr,
                tf.where(mask, tf.constant(1, tf.int64), 0),
                tf.where(mask, length - 1, shape),
            ),
            fill,
        ],
        axis=axis,
    )

    # TODO: validate lower == upper except for axis
    lower = tf.where(tf.math.equal(curr - prev, 1))
    upper = tf.where(tf.math.equal(next - curr, -1))

    # Fix values with -1 (where indices is not available)
    start = core_ops.io_order_indices(lower, shape, axis)
    start = tf.where(tf.math.greater_equal(start, 0), start, length)
    stop = core_ops.io_order_indices(upper, shape, axis)
    stop = tf.where(tf.math.greater_equal(stop, 0), stop + 1, length)

    return tf.stack([start, stop], axis=axis + 1)


def trim(input, axis, epsilon, name=None):
    """
    Trim the noise from beginning and end of the audio.

    Args:
      input: An audio Tensor.
      axis: The axis to trim.
      epsilon: The max value to be considered as noise.
      name: A name for the operation (optional).
    Returns:
      A tensor of start and stop with shape `[..., 2, ...]`.
    """
    shape = tf.shape(input, out_type=tf.int64)
    length = shape[axis]

    nonzero = tf.math.greater(input, epsilon)
    check = tf.reduce_any(nonzero, axis=axis)

    forward = tf.cast(nonzero, tf.int8)
    reverse = tf.reverse(forward, [axis])

    start = tf.where(check, tf.argmax(forward, axis=axis), length)
    stop = tf.where(check, tf.argmax(reverse, axis=axis), tf.constant(0, tf.int64))
    stop = length - stop

    return tf.stack([start, stop], axis=axis)


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
    input = tf.convert_to_tensor(input)
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
    return tf.where(condition, tf.cast(0, input.dtype), input)


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
    input = tf.convert_to_tensor(input)
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
    return tf.where(condition, tf.cast(0, input.dtype), input)


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

def _get_sinc_resample_kernel(rate_in, rate_out, lowpass_filter_width):
    assert lowpass_filter_width > 0
    base_freq = min(rate_in, rate_out)
    # This will perform antialiasing filtering by removing the highest frequencies.
    # At first I thought I only needed this when downsampling, but when upsampling
    # you will get edge artifacts without this, as the edge is equivalent to zero padding,
    # which will add high freq artifacts.
    base_freq *= 0.99

    # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
    # using the sinc interpolation formula:
    #   x(t) = sum_i x[i] sinc(pi * rate_in * (i / rate_in - t))
    # We can then sample the function x(t) with a different sample rate:
    #    y[j] = x(j / rate_out)
    # or,
    #    y[j] = sum_i x[i] sinc(pi * rate_in * (i / rate_in - j / rate_out))

    # We see here that y[j] is the convolution of x[i] with a specific filter, for which
    # we take an FIR approximation, stopping when we see at least `lowpass_filter_width` zeros crossing.
    # But y[j+1] is going to have a different set of weights and so on, until y[j + rate_out].
    # Indeed:
    # y[j + rate_out] = sum_i x[i] sinc(pi * rate_in * ((i / rate_in - (j + rate_out) / rate_out))
    #                 = sum_i x[i] sinc(pi * rate_in * ((i - rate_in) / rate_in - j / rate_out))
    #                 = sum_i x[i + rate_in] sinc(pi * rate_in * (i / rate_in - j / rate_out))
    # so y[j+rate_out] uses the same filter as y[j], but on a shifted version of x by `rate_in`.
    # This will explain the F.conv1d after, with a stride of rate_in.
    width = math.ceil(lowpass_filter_width * rate_in / base_freq)
    # If rate_in is still big after GCD reduction, most filters will be very unbalanced, i.e.,
    # they will have a lot of almost zero values to the left or to the right...
    # There is probably a way to evaluate those filters more efficiently, but this is kept for
    # future work.
    idx = tf.range(-width, width + rate_in,dtype=tf.float32)
    idx=tf.repeat(tf.expand_dims(idx,axis=-1),rate_out,axis=-1)
    aux_i=tf.expand_dims(tf.range(rate_out,dtype=tf.float32),axis=0)
    kernels = (-aux_i / rate_out + idx / rate_in) * base_freq
        
    kernels = tf.clip_by_value(kernels,-lowpass_filter_width, lowpass_filter_width)
    kernels *= math.pi

    window = tf.math.cos(kernels / lowpass_filter_width / 2)**2
    kernels = tf.where(kernels == 0, tf.ones_like(kernels), tf.math.sin(kernels) / kernels)
    kernels*=window
    
    scale = base_freq / rate_in
    return tf.expand_dims(kernels,axis=1)*scale, width

def resample(input, rate_in, rate_out, lowpass_filter_width= 6):
    """Resamples the waveform at the new frequency. This matches Kaldi’s OfflineFeatureTpl ResampleWaveform which uses a LinearResample (resample a signal at linearly spaced intervals to upsample/downsample a signal). LinearResample (LR) means that the output signal is at linearly spaced intervals (i.e the output signal has a frequency of rate_out). It uses sinc/bandlimited interpolation to upsample/downsample the signal.

    Args:
      waveform: A 1-D (`[samples]`) or 2-D (`[samples, channels]`) or 3-D (`[batch, samples, channels]`) `Tensor` of type `float`. Audio input.
      rate_in: The rate of the audio input.
      rate_out: The rate of the audio output.
      lowpass_filter_width:  Controls the sharpness of the filter, more == sharper but less efficient. We suggest around 4 to 10 for normal use. (Default: 6)

    Returns:
      output: Resampled audio.
    """
    waveform = input

    if rate_in==rate_out:
        return waveform
    
    rate_in = int(rate_in)
    rate_out = int(rate_out)
    gcd = math.gcd(rate_in, rate_out)
    rate_in = rate_in // gcd
    rate_out = rate_out // gcd

    kernel, width = _get_sinc_resample_kernel(rate_in, rate_out, lowpass_filter_width)

    ori_shape=waveform.shape
    ori_shape_len=len(ori_shape)
    if ori_shape_len==1:
        waveform = tf.expand_dims(waveform,axis=0)
    elif ori_shape_len==2:
        waveform = tf.transpose(waveform,[1,0])
    elif ori_shape_len==3:
        waveform = tf.transpose(waveform,[0,2,1])
        waveform = tf.reshape(waveform,[ori_shape[0]*ori_shape[2],ori_shape[1]])

    waveform = tf.expand_dims(waveform,axis=-1)

    num_wavs, length, _ = waveform.shape
    
    waveform = tf.pad(waveform,[[0,0],[width, width + rate_in],[0,0]])
    resampled = tf.nn.conv1d(waveform, kernel, stride=rate_in,padding='VALID')
    resampled = tf.reshape(resampled,[num_wavs, -1])
    target_length = int(math.ceil(rate_out * length / rate_in))
    if ori_shape_len==1:
        return resampled[0, :target_length]
    elif ori_shape_len==2:
        return tf.transpose(resampled[:, :target_length],[1,0])
    elif ori_shape_len==3:
        return tf.transpose(
            tf.reshape(
                resampled[:, :target_length],[ori_shape[0],ori_shape[2],target_length]
            ),[0,2,1]
        )


def decode_wav(
    input, shape=None, dtype=None, name=None
):  # pylint: disable=redefined-builtin
    """Decode WAV audio from input string.

    Args:
      input: A string `Tensor` of the audio input.
      shape: The shape of the audio.
      dtype: The data type of the audio, only
        tf.uint8, tf.int16, tf.int32 and tf.float32 are supported.
      name: A name for the operation (optional).

    Returns:
      output: Decoded audio.
    """
    if shape is None:
        shape = tf.constant([-1, -1], tf.int64)
    assert (
        dtype is not None
    ), "dtype (tf.uint8/tf.int16/tf.int32/tf.float32) must be provided"
    return core_ops.io_audio_decode_wav(input, shape=shape, dtype=dtype, name=name)


def encode_wav(input, rate, name=None):  # pylint: disable=redefined-builtin
    """Encode WAV audio into string.

    Args:
      input: A `Tensor` of the audio input.
      rate: The sample rate of the audio.
      name: A name for the operation (optional).

    Returns:
      output: Encoded audio.
    """
    return core_ops.io_audio_encode_wav(input, rate, name=name)


def decode_flac(
    input, shape=None, dtype=None, name=None
):  # pylint: disable=redefined-builtin
    """Decode Flac audio from input string.

    Args:
      input: A string `Tensor` of the audio input.
      shape: The shape of the audio.
      dtype: The data type of the audio, only
        tf.uint8, tf.int16 and tf.int32 are supported.
      name: A name for the operation (optional).

    Returns:
      output: Decoded audio.
    """
    if shape is None:
        shape = tf.constant([-1, -1], tf.int64)
    assert dtype is not None, "dtype (tf.uint8/tf.int16/tf.int32) must be provided"
    return core_ops.io_audio_decode_flac(input, shape=shape, dtype=dtype, name=name)


def encode_flac(input, rate, name=None):  # pylint: disable=redefined-builtin
    """Encode Flac audio into string.

    Args:
      input: A `Tensor` of the audio input.
      rate: The sample rate of the audio.
      name: A name for the operation (optional).

    Returns:
      output: Encoded audio.
    """
    return core_ops.io_audio_encode_flac(input, rate, name=name)


def decode_vorbis(input, shape=None, name=None):  # pylint: disable=redefined-builtin
    """Decode Ogg(Vorbis) audio from input string.

    Args:
      input: A string `Tensor` of the audio input.
      shape: The shape of the audio.
      name: A name for the operation (optional).

    Returns:
      output: Decoded audio as tf.float32.
    """
    if shape is None:
        shape = tf.constant([-1, -1], tf.int64)
    return core_ops.io_audio_decode_vorbis(input, shape=shape, name=name)


def encode_vorbis(input, rate, name=None):  # pylint: disable=redefined-builtin
    """Encode Ogg(Vorbis) audio into string.

    Args:
      input: A `Tensor` of the audio input.
      rate: The sample rate of the audio.
      name: A name for the operation (optional).

    Returns:
      output: Encoded audio.
    """
    return core_ops.io_audio_encode_vorbis(input, rate, name=name)


def decode_mp3(input, shape=None, name=None):  # pylint: disable=redefined-builtin
    """Decode MP3 audio from input string.

    Args:
      input: A string `Tensor` of the audio input.
      shape: The shape of the audio.
      name: A name for the operation (optional).

    Returns:
      output: Decoded audio as tf.float32.
    """
    if shape is None:
        shape = tf.constant([-1, -1], tf.int64)
    return core_ops.io_audio_decode_mp3(input, shape=shape, name=name)


def encode_mp3(input, rate, name=None):  # pylint: disable=redefined-builtin
    """Encode MP3 audio into string.

    Args:
      input: A `Tensor` of the audio input.
      rate: The sample rate of the audio.
      name: A name for the operation (optional).

    Returns:
      output: Encoded audio.
    """
    return core_ops.io_audio_encode_mp3(input, rate, name=name)


def decode_aac(input, shape=None, name=None):  # pylint: disable=redefined-builtin
    """Decode MP4 (AAC) audio from input string.

    Args:
      input: A string `Tensor` of the audio input.
      shape: The shape of the audio.
      name: A name for the operation (optional).

    Returns:
      output: Decoded audio as tf.float32.
    """
    if shape is None:
        shape = tf.constant([-1, -1], tf.int64)
    if sys.platform == "linux":
        try:
            from tensorflow_io.core.python.ops import (  # pylint: disable=import-outside-toplevel,unused-import
                ffmpeg_ops,
            )
        except NotImplementedError:
            pass
    return core_ops.io_audio_decode_aac(input, shape=shape, name=name)


def encode_aac(input, rate, name=None):  # pylint: disable=redefined-builtin
    """Encode MP4(AAC) audio into string.

    Args:
      input: A `Tensor` of the audio input.
      rate: The sample rate of the audio.
      name: A name for the operation (optional).

    Returns:
      output: Encoded audio.
    """
    if sys.platform == "linux":
        try:
            from tensorflow_io.core.python.ops import (  # pylint: disable=import-outside-toplevel,unused-import
                ffmpeg_ops,
            )
        except NotImplementedError:
            pass
    return core_ops.io_audio_encode_aac(input, rate, name=name)


class AudioIOTensor:
    """AudioIOTensor"""

    # =============================================================================
    # Constructor
    # =============================================================================
    def __init__(self, filename, dtype=None):
        with tf.name_scope("AudioIOTensor"):
            if not tf.executing_eagerly():
                assert dtype is not None, "dtype must be provided in graph mode"
            resource = core_ops.io_audio_readable_init(filename)
            if tf.executing_eagerly():
                shape, dtype, rate = core_ops.io_audio_readable_spec(resource)
                dtype = tf.as_dtype(dtype.numpy())
            else:
                shape, _, rate = core_ops.io_audio_readable_spec(resource)
            self._resource = resource
            self._shape = shape
            self._dtype = dtype
            self._rate = rate
            super().__init__()

    # =============================================================================
    # Accessors
    # =============================================================================

    @property
    def shape(self):
        """Returns the `TensorShape` that represents the shape of the tensor."""
        return self._shape

    @property
    def dtype(self):
        """Returns the `dtype` of elements in the tensor."""
        return self._dtype

    @property
    def rate(self):
        """The sample `rate` of the audio stream"""
        return self._rate

    # =============================================================================
    # String Encoding
    # =============================================================================
    def __repr__(self):
        return "<AudioIOTensor: shape={}, dtype={}, rate={}>".format(
            self.shape, self.dtype, self.rate
        )

    # =============================================================================
    # Tensor Type Conversions
    # =============================================================================

    def to_tensor(self):
        """Converts this `IOTensor` into a `tf.Tensor`.

        Args:
          name: A name prefix for the returned tensors (optional).

        Returns:
          A `Tensor` with value obtained from this `IOTensor`.
        """
        return core_ops.io_audio_readable_read(self._resource, 0, -1, dtype=self._dtype)

    # =============================================================================
    # Indexing and slicing
    # =============================================================================
    def __getitem__(self, key):
        """Returns the specified piece of this IOTensor."""
        # always convert to tuple to process
        if not isinstance(key, tuple):
            key = tuple([key])
        # get the start and stop of each element
        indices = [
            (k.start, k.stop) if isinstance(k, slice) else (k, k + 1) for k in key
        ]
        # get the start and stop, and use 0 (start) and -1 (stop) if needed
        indices = list(zip(*indices))
        start = [0 if e is None else e for e in indices[0]]
        stop = [-1 if e is None else e for e in indices[1]]

        item = core_ops.io_audio_readable_read(
            self._resource, start=start, stop=stop, dtype=self._dtype
        )

        # in case certain dimension is not slice, then this dimension will need to
        # collapse as `0`, otherwise `:` or `slice(None, None, None)`
        indices = [slice(None) if isinstance(k, slice) else 0 for k in key]

        return item.__getitem__(indices)

    def __len__(self):
        """Returns the total number of items of this IOTensor."""
        return self._shape[0]


class AudioIODataset(tf.data.Dataset):
    """AudioIODataset"""

    def __init__(self, filename, dtype=None):
        """AudioIODataset."""
        with tf.name_scope("AudioIODataset"):
            if not tf.executing_eagerly():
                assert dtype is not None, "dtype must be provided in graph mode"
            resource = core_ops.io_audio_readable_init(filename)
            if tf.executing_eagerly():
                shape, dtype, _ = core_ops.io_audio_readable_spec(resource)
                dtype = tf.as_dtype(dtype.numpy())
            else:
                shape, _, _ = core_ops.io_audio_readable_spec(resource)

            capacity = 1024  # kwargs.get("capacity", 4096)

            self._resource = resource
            dataset = tf.data.Dataset.range(0, shape[0], capacity)
            dataset = dataset.map(
                lambda index: core_ops.io_audio_readable_read(
                    resource, index, index + capacity, dtype=dtype
                )
            )
            dataset = dataset.apply(
                tf.data.experimental.take_while(lambda v: tf.greater(tf.shape(v)[0], 0))
            )
            dataset = dataset.unbatch()
            self._dataset = dataset
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
