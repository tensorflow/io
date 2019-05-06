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
import pytest

import tensorflow
tensorflow.compat.v1.disable_eager_execution()

if hasattr(tensorflow, "audio"):
  from tensorflow import audio         # pylint: disable=wrong-import-position
else:
  from tensorflow.contrib.framework.python.ops import audio_ops as audio # pylint: disable=wrong-import-position
from tensorflow import errors          # pylint: disable=wrong-import-position

import tensorflow_io.audio as audio_io # pylint: disable=wrong-import-position

audio_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_audio", "mono_10khz.wav")

def test_audio_dataset():
  """Test Audio Dataset"""
  with open(audio_path, 'rb') as f:
    wav_contents = f.read()
  audio_p = audio.decode_wav(wav_contents)
  with tensorflow.compat.v1.Session() as sess:
    audio_v = sess.run(audio_p).audio

  f = lambda x: float(x) / (1 << 15)

  dataset = audio_io.WAVDataset([audio_path])
  iterator = dataset.make_initializable_iterator()
  init_op = iterator.initializer
  get_next = iterator.get_next()
  with tensorflow.compat.v1.Session() as sess:
    sess.run(init_op)
    for i in range(audio_v.shape[0]):
      v = sess.run(get_next)
      assert audio_v[i] == f(v)
    with pytest.raises(errors.OutOfRangeError):
      sess.run(get_next)

  dataset = audio_io.WAVDataset([audio_path], batch=2)
  iterator = dataset.make_initializable_iterator()
  init_op = iterator.initializer
  get_next = iterator.get_next()
  with tensorflow.compat.v1.Session() as sess:
    sess.run(init_op)
    for i in range(0, audio_v.shape[0], 2):
      v = sess.run(get_next)
      assert audio_v[i] == f(v[0])
      assert audio_v[i + 1] == f(v[1])
    with pytest.raises(errors.OutOfRangeError):
      sess.run(get_next)
