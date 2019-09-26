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
"""test ffmpeg dataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pytest
import numpy as np

import tensorflow as tf
if not (hasattr(tf, "version") and tf.version.VERSION.startswith("2.")):
  tf.compat.v1.enable_eager_execution()
if sys.platform == "darwin":
  pytest.skip("video is not supported on macOS yet", allow_module_level=True)
import tensorflow_io as tfio  # pylint: disable=wrong-import-position
import tensorflow_io.ffmpeg as ffmpeg_io  # pylint: disable=wrong-import-position

video_path = "file://" + os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_video", "small.mp4")
def test_video_dataset():
  """test_video_dataset"""
  num_repeats = 2

  video_dataset = ffmpeg_io.VideoDataset([video_path]).repeat(num_repeats)

  i = 0
  for v in video_dataset:
    assert v.shape == (320, 560, 3)
    i += 1
  assert i == 166 * num_repeats

audio_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_audio", "mono_10khz.wav")

def test_audio_dataset():
  """Test Audio Dataset"""
  with open(audio_path, 'rb') as f:
    wav_contents = f.read()
  audio_v = tf.audio.decode_wav(wav_contents)

  f = lambda x: float(x) / (1 << 15)

  audio_dataset = ffmpeg_io.AudioDataset(audio_path)
  i = 0
  for v in audio_dataset:
    assert audio_v.audio[i].numpy() == f(v.numpy())
    i += 1
  assert i == 5760

  audio_dataset = ffmpeg_io.AudioDataset(audio_path).batch(2)
  i = 0
  for v in audio_dataset:
    assert audio_v.audio[i].numpy() == f(v[0].numpy())
    assert audio_v.audio[i + 1].numpy() == f(v[1].numpy())
    i += 2
  assert i == 5760

def test_ffmpeg_io_tensor_audio():
  """test_ffmpeg_io_tensor_audio"""
  audio = tfio.IOTensor.from_audio(audio_path)
  ffmpeg = tfio.IOTensor.from_ffmpeg(audio_path)
  ffmpeg = ffmpeg('a:0')
  assert audio.dtype == ffmpeg.dtype
  assert audio.rate == ffmpeg.rate
  assert audio.shape[1] == ffmpeg.shape[1]
  assert np.all(audio[0:5760].numpy() == ffmpeg[0:5760].numpy())
  assert len(audio) == len(ffmpeg)

  audio_24bit_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "example_0.5s.wav")
  audio_24bit = tfio.IOTensor.from_ffmpeg(audio_24bit_path)
  assert audio_24bit('a:0').shape.as_list() == [None, 2]
  assert audio_24bit('a:0').dtype == tf.int32
  assert audio_24bit('a:0').rate == 44100

# Disable as the mkv file is large. Run locally
# by pulling test file while is located in:
# https://github.com/Matroska-Org/matroska-test-files/blob/master/test_files/test5.mkv
def _test_ffmpeg_io_tensor_mkv():
  """test_ffmpeg_io_tensor_mkv"""
  mkv_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_video", "test5.mkv")
  mkv = tfio.IOTensor.from_ffmpeg(mkv_path)
  assert mkv('a:0').shape.as_list() == [None, 2]
  assert mkv('a:0').dtype == tf.float32
  assert mkv('a:0').rate == 48000
  assert mkv('s:0').shape.as_list() == [None]
  assert mkv('s:0').dtype == tf.string
  assert mkv('s:0')[0] == ['...the colossus of Rhodes!\r\n']
  assert mkv('s:0')[1] == ['No!\r\n']
  assert mkv('s:0')[2] == [
      'The colossus of Rhodes\\Nand it is here just for you Proog.\r\n']
  assert mkv('s:0')[3] == ['It is there...\r\n']
  assert mkv('s:0')[4] == ["I'm telling you,\\NEmo...\r\n"]

  video = tfio.IOTensor.from_ffmpeg(video_path)
  assert video('v:0').shape.as_list() == [None, 320, 560, 3]
  assert video('v:0').dtype == tf.uint8
  assert len(video('v:0')) == 166
  assert video('v:0').to_tensor().shape == [166, 320, 560, 3]

def test_ffmpeg_decode_video():
  """test_ffmpeg_decode_video"""
  content = tf.io.read_file(video_path)
  video = ffmpeg_io.decode_video(content, 0)
  assert video.shape == [166, 320, 560, 3]
  assert video.dtype == tf.uint8
