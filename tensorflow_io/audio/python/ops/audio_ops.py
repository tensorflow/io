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
  start = kwargs.get("start", 0)
  count = kwargs.get("count", None)
  memory = kwargs.get("memory", "")
  if count is None and spec.shape[0] is not None:
    count = spec.shape[0] - start
  if count is None:
    count = -1
  return core_ops.read_wav(
      filename,
      start=start, count=count, dtype=spec.dtype,
      memory=memory)

class WAVDataset(data_ops.BaseDataset):
  """A WAV Dataset"""

  def __init__(self, filename, **kwargs):
    """Create a WAVDataset.

    Args:
      filename: A string containing filename.
    """
    if not tf.executing_eagerly():
      count = kwargs.get("count")
      dtype = kwargs.get("dtype")
      shape = kwargs.get("shape")
    else:
      spec, _ = list_wav_info(filename)
      count = spec.shape[0]
      dtype = spec.dtype
      shape = tf.TensorShape([None])

    # capacity is the rough count for each chunk in dataset
    capacity = kwargs.get("capacity", 65536)
    entry_start = list(range(0, count, capacity))
    entry_count = [min(capacity, count - start) for start in entry_start]
    dataset = data_ops.BaseDataset.from_tensor_slices(
        (tf.constant(entry_start, tf.int64), tf.constant(entry_count, tf.int64))
    ).map(lambda start, count: core_ops.read_wav(
        filename, start, count, dtype=dtype, memory=""))
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
