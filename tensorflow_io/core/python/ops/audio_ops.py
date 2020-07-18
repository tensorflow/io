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


def resample(input, rate_in, rate_out, name=None):  # pylint: disable=redefined-builtin
    """Resample audio.

    Args:
      input: A 1-D (`[samples]`) or 2-D (`[samples, channels]`) `Tensor` of
        type `int16` or `float`. Audio input.
      rate_in: The rate of the audio input.
      rate_out: The rate of the audio output.
      name: A name for the operation (optional).

    Returns:
      output: Resampled audio.
    """
    rank = tf.rank(input)

    input = tf.cond(
        tf.math.equal(rank, 1), lambda: tf.expand_dims(input, -1), lambda: input
    )
    value = core_ops.io_audio_resample(
        input, rate_in=rate_in, rate_out=rate_out, name=name
    )
    return tf.cond(
        tf.math.equal(rank, 1), lambda: tf.squeeze(value, [-1]), lambda: value
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
