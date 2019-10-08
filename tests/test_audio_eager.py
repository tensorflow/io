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
import numpy as np

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
import tensorflow_io as tfio # pylint: disable=wrong-import-position

audio_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_audio", "mono_10khz.wav")

def test_from_tensor():
  """test_from_tensor"""
  audio_data = tfio.IOTensor.from_audio(audio_path)
  numpy_data = audio_data.to_tensor().numpy()

  new_tensor = tfio.IOTensor.from_tensor(numpy_data)
  # The behavior of from_audio and from_tesnor should match
  for i, audio_value in enumerate(audio_data):
    assert new_tensor[i].numpy() == audio_value.numpy()
def test_audio_dataset():
  """Test Audio Dataset"""
  with open(audio_path, 'rb') as f:
    wav_contents = f.read()
  audio_v = tf.audio.decode_wav(wav_contents)

  f = lambda x: float(x) / (1 << 15)

  audio_dataset = tfio.IODataset.from_audio(audio_path)
  i = 0
  for v in audio_dataset:
    assert audio_v.audio[i].numpy() == f(v.numpy())
    i += 1
  assert i == 5760

  audio_dataset = tfio.IODataset.from_audio(audio_path).batch(2)
  i = 0
  for v in audio_dataset:
    assert audio_v.audio[i].numpy() == f(v[0].numpy())
    assert audio_v.audio[i + 1].numpy() == f(v[1].numpy())
    i += 2
  assert i == 5760

  samples = tfio.IOTensor.from_audio(audio_path)
  assert samples.dtype == tf.int16
  assert samples.shape == [5760, 1]
  assert samples.rate == audio_v.sample_rate.numpy()

  audio_24bit_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "example_0.5s.wav")
  # raw was geenrated from:
  # $ sox example_0.5s.wav example_0.5s.s32
  audio_24bit_raw_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "example_0.5s.s32")
  expected = np.fromfile(audio_24bit_raw_path, np.int32)
  expected = np.reshape(expected, [22050, 2])

  samples = tfio.IOTensor.from_audio(audio_24bit_path)
  assert samples.dtype == tf.int32
  assert samples.shape == [22050, 2]
  assert samples.rate == 44100
  assert np.all(samples.to_tensor().numpy() == expected)
