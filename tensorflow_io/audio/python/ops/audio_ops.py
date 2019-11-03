# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Audio Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops

warnings.warn(
    "The tensorflow_io.audio.WAVDataset is "
    "deprecated. Please look for tfio.IOTensor.from_audio "
    "for reading WAV files into tensorflow.",
    DeprecationWarning)


class AudioDataset(data_ops.Dataset):
  """A Audio File Dataset that reads the audio file."""

  def __init__(self, filename, batch=None):
    """Create a `AudioDataset`.
    Args:
      filename: A `tf.string` tensor containing one or more filenames.
    """
    batch = 0 if batch is None else batch
    dtypes = [tf.int16]
    shapes = [
        tf.TensorShape([None])] if batch == 0 else [
            tf.TensorShape([None, None])]
    super(AudioDataset, self).__init__(
        ffmpeg_ops.io_audio_dataset,
        ffmpeg_ops.io_audio_input(filename),
        batch, dtypes, shapes)
