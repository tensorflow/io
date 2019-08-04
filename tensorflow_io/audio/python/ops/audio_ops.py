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
from tensorflow_io.core.python.ops import core_ops

def list_wav_info(filename, **kwargs):
  """list_wav_info"""
  if not tf.executing_eagerly():
    raise NotImplementedError("list_wav_info only support eager mode")
  memory = kwargs.get("memory", "")
  dtype, shape, rate = core_ops.list_wav_info(
      filename, memory=memory)
  return tf.TensorSpec(shape.numpy(), dtype.numpy().decode()), rate

def read_wav(filename, spec, **kwargs):
  """read_wav"""
  memory = kwargs.get("memory", "")
  start = kwargs.get("start", 0)
  stop = kwargs.get("stop", None)
  if stop is None and spec.shape[0] is not None:
    stop = spec.shape[0] - start
  if stop is None:
    stop = -1
  return core_ops.read_wav(
      filename, memory=memory,
      start=start, stop=stop, dtype=spec.dtype)

class WAVDataset(data_ops.BaseDataset):
  """A WAV Dataset"""

  def __init__(self, filename, **kwargs):
    """Create a WAVDataset.

    Args:
      filename: A string containing filename.
    """
    if not tf.executing_eagerly():
      start = kwargs.get("start")
      stop = kwargs.get("stop")
      dtype = kwargs.get("dtype")
      shape = kwargs.get("shape")
    else:
      spec, _ = list_wav_info(filename)
      start = 0
      stop = spec.shape[0]
      dtype = spec.dtype
      shape = tf.TensorShape(
          [dim if i != 0 else None for i, dim in enumerate(
              spec.shape.as_list())])

    # capacity is the rough count for each chunk in dataset
    capacity = kwargs.get("capacity", 65536)
    entry_start = list(range(start, stop, capacity))
    entry_stop = entry_start[1:] + [stop]
    dataset = data_ops.BaseDataset.from_tensor_slices(
        (tf.constant(entry_start, tf.int64), tf.constant(entry_stop, tf.int64))
    ).map(lambda start, stop: core_ops.read_wav(
        filename, memory="", start=start, stop=stop, dtype=dtype))
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
