# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Test Audio Dataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io.audio as audio_io # pylint: disable=wrong-import-position

audio_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_audio", "mono_10khz.wav")

def test_audio_dataset():
  """Test Audio Dataset"""
  with open(audio_path, 'rb') as f:
    wav_contents = f.read()
  audio_v = tf.audio.decode_wav(wav_contents)

  f = lambda x: float(x) / (1 << 15)

  for capacity in [10, 100, 500]:
    audio_dataset = audio_io.WAVDataset(audio_path, capacity=capacity).apply(
        tf.data.experimental.unbatch()).map(tf.squeeze)
    i = 0
    for v in audio_dataset:
      assert audio_v.audio[i].numpy() == f(v.numpy())
      i += 1
    assert i == 5760

  for capacity in [10, 100, 500]:
    audio_dataset = audio_io.WAVDataset(audio_path, capacity=capacity).apply(
        tf.data.experimental.unbatch()).batch(2).map(tf.squeeze)
    i = 0
    for v in audio_dataset:
      assert audio_v.audio[i].numpy() == f(v[0].numpy())
      assert audio_v.audio[i + 1].numpy() == f(v[1].numpy())
      i += 2
    assert i == 5760

  spec, rate = audio_io.list_wav_info(audio_path)
  assert spec.dtype == tf.int16
  assert spec.shape == [5760, 1]
  assert rate.numpy() == audio_v.sample_rate.numpy()

  samples = audio_io.read_wav(audio_path, spec)
  assert samples.dtype == tf.int16
  assert samples.shape == [5760, 1]
