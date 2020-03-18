# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test Audio"""

import os
import pytest
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio

@pytest.fixture(name="fixture_lookup")
def fixture_lookup_func(request):
  def _fixture_lookup(name):
    return request.getfixturevalue(name)
  return _fixture_lookup

@pytest.fixture(name="resample", scope="module")
def fixture_resample():
  """fixture_resample"""
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.wav")
  audio = tf.audio.decode_wav(tf.io.read_file(path))
  value = audio.audio * (1 << 15)
  value = tf.cast(value, tf.int16)

  expected_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain-4410-quality-1.wav")
  expected_audio = tf.audio.decode_wav(tf.io.read_file(expected_path))
  expected_value = expected_audio.audio * (1 << 15)
  expected_value = tf.cast(expected_value, tf.int16)

  args = value
  func = lambda e: tfio.experimental.audio.resample(value, 44100, 4410, 1)
  expected = expected_value

  return args, func, expected

@pytest.fixture(name="decode_wav", scope="module")
def fixture_decode_wav():
  """fixture_decode_wav"""
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.wav")
  content = tf.io.read_file(path)

  audio = tf.audio.decode_wav(tf.io.read_file(path))
  value = audio.audio * (1 << 15)
  value = tf.cast(value, tf.int16)

  args = content
  func = lambda e: tfio.experimental.audio.decode_wav(content, dtype=tf.int16)
  expected = value

  return args, func, expected

# By default, operations runs in eager mode,
# Note as of now shape inference is skipped in eager mode
@pytest.mark.parametrize(
    ("io_data_fixture"),
    [
        pytest.param("resample"),
        pytest.param("decode_wav"),
    ],
    ids=[
        "resample",
        "decode_wav",
    ],
)
def test_audio_ops(fixture_lookup, io_data_fixture):
  """test_audio_ops"""
  args, func, expected = fixture_lookup(io_data_fixture)

  entries = func(args)
  assert np.array_equal(entries, expected)

def test_general_decode_mp3():
  """test generic audio decode for mp3"""
  expected_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "sine440_cbr128.mp3")
  contents = tf.io.read_file(expected_path)

  samples, _ = tfio.audio.decode(contents)

  assert tf.rank(samples) == 2

def test_general_decode_wav():
  """test generic audio decode for wav"""
  expected_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "mono_10khz.wav")
  contents = tf.io.read_file(expected_path)

  with pytest.raises(tf.errors.InvalidArgumentError):
    samples, _ = tfio.audio.decode(contents)

def test_general_decode_flac():
  """test generic audio decode for flac"""
  expected_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.flac")
  contents = tf.io.read_file(expected_path)

  with pytest.raises(tf.errors.InvalidArgumentError):
    samples, _ = tfio.audio.decode(contents)

def test_general_decode_ogg():
  """test generic audio decode for ogg"""
  expected_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.ogg")
  contents = tf.io.read_file(expected_path)

  with pytest.raises(tf.errors.InvalidArgumentError):
    samples, _ = tfio.audio.decode(contents)

def test_general_decode_unsupported():
  """test generic audio decode for unsupported format"""
  contents = b""

  with pytest.raises(tf.errors.InvalidArgumentError):
    samples, _ = tfio.audio.decode(contents)

def test_decode_mp3():
  """test standard decoding of a mono MP3 file"""
  expected_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "sine440_cbr128.mp3")
  contents = tf.io.read_file(expected_path)

  samples, sample_rate = tfio.audio.decode_mp3(contents)

  assert sample_rate == 44100
  assert np.array_equal(samples.shape, [1, 44100])

def test_decode_mp3_mono2stereo():
  """test MP3 decoding with conversion of mono to stereo"""
  expected_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "sine440_cbr128.mp3")
  contents = tf.io.read_file(expected_path)

  samples, _ = tfio.audio.decode_mp3(contents, desired_channels=2)

  assert np.array_equal(samples.shape, [2, 44100])
  assert np.array_equal(samples[0], samples[1])

def test_decode_mp3_padded():
  """test MP3 decoding with padding to a fixed sample count"""
  expected_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "sine440_cbr128.mp3")
  contents = tf.io.read_file(expected_path)

  samples, _ = tfio.audio.decode_mp3(contents)
  samples_padded, _ = tfio.audio.decode_mp3(contents, desired_samples=44200)

  assert np.array_equal(samples_padded.shape, [1, 44200])
  assert np.array_equal(samples_padded[:, :44100], samples)
  assert np.array_equal(samples_padded[:, 44100:], np.zeros((1, 100)))

def test_decode_mp3_truncated():
  """test MP3 decoding with truncating to a fixed sample count"""
  expected_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "sine440_cbr128.mp3")
  contents = tf.io.read_file(expected_path)

  samples, _ = tfio.audio.decode_mp3(contents)
  samples_truncated, _ = tfio.audio.decode_mp3(contents, desired_samples=10)

  assert np.array_equal(samples_truncated.shape, [1, 10])
  assert np.array_equal(samples_truncated, samples[:, :10])

# A tf.data pipeline runs in graph mode and shape inference is invoked.
@pytest.mark.parametrize(
    ("io_data_fixture"),
    [
        pytest.param("resample"),
        pytest.param("decode_wav"),
    ],
    ids=[
        "resample",
        "decode_wav",
    ],
)
def test_audio_ops_in_graph(fixture_lookup, io_data_fixture):
  """test_audio_ops_in_graph"""
  args, func, expected = fixture_lookup(io_data_fixture)

  dataset = tf.data.Dataset.from_tensor_slices([args])
  dataset = dataset.map(func)
  entries = list(dataset)
  assert len(entries) == 1
  entries = entries[0]
