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

    TODO: Support audio with channel > 1.

    Args:
      input: An 1-D audio signal Tensor.
      nfft: Size of FFT.
      window: Size of sindow.
      stride: size of hope between windows.
      name: A name for the operation (optional).

    Returns:
      A tensor of spectrogram.
    """

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
