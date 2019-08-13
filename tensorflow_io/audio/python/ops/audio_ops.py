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

import tensorflow as tf
from tensorflow_io.core.python.ops import data_ops
from tensorflow_io.core.python.ops import io_tensor

class WAVDataset(data_ops.BaseDataset):
  """A WAV Dataset"""

  def __init__(self, filename, batch=None, **kwargs):
    """Create a WAVDataset.

    Args:
      filename: A string containing filename.
    """
    batch = 0 if batch is None else batch
    if not tf.executing_eagerly():
      raise NotImplementedError("WAVDataset only support eager mode")

    self._wav = io_tensor.IOTensor.from_audio(filename)

    dtype = self._wav.dtype
    shape = self._wav.shape[1:]
    start = 0
    stop = self._wav.shape[0]

    # capacity is the rough count for each chunk in dataset
    capacity = kwargs.get("capacity", 65536)
    entry_start = list(range(start, stop, capacity))
    entry_stop = entry_start[1:] + [stop]
    dataset = data_ops.BaseDataset.from_tensor_slices(
        (tf.constant(entry_start, tf.int64), tf.constant(entry_stop, tf.int64))
    ).map(lambda start, stop: self._wav.__getitem__(slice(start, stop)))

    dataset = dataset.apply(tf.data.experimental.unbatch())
    if batch != 0:
      dataset = dataset.batch(batch)
      shape = tf.TensorShape([None]).concatenate(shape)

    self._dataset = dataset
    super(WAVDataset, self).__init__(
        self._dataset._variant_tensor, [dtype], [shape]) # pylint: disable=protected-access

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
        ffmpeg_ops.audio_dataset,
        ffmpeg_ops.audio_input(filename),
        batch, dtypes, shapes)
