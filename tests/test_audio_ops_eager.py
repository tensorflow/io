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
  func = lambda e: tfio.experimental.audio.resample(e, 44100, 4410, 1)
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
  func = lambda e: tfio.experimental.audio.decode_wav(e, dtype=tf.int16)
  expected = value

  return args, func, expected

@pytest.fixture(name="decode_flac", scope="module")
def fixture_decode_flac():
  """fixture_decode_flac"""
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.flac")
  content = tf.io.read_file(path)

  wav_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.wav")
  audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
  value = audio.audio * (1 << 15)
  value = tf.cast(value, tf.int16)

  args = content
  func = tfio.experimental.audio.decode_flac
  expected = value

  return args, func, expected

@pytest.fixture(name="encode_flac", scope="module")
def fixture_encode_flac():
  """fixture_encode_flac"""
  wav_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.wav")
  audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
  value = audio.audio * (1 << 15)
  value = tf.cast(value, tf.int16)

  args = value
  def func(e):
    v = tfio.experimental.audio.encode_flac(e, rate=44100)
    return tfio.experimental.audio.decode_flac(v)
  expected = value

  return args, func, expected

# By default, operations runs in eager mode,
# Note as of now shape inference is skipped in eager mode
@pytest.mark.parametrize(
    ("io_data_fixture"),
    [
        pytest.param("resample"),
        pytest.param("decode_wav"),
        pytest.param("decode_flac"),
        pytest.param("encode_flac"),
    ],
    ids=[
        "resample",
        "decode_wav",
        "decode_flac",
        "encode_flac",
    ],
)
def test_audio_ops(fixture_lookup, io_data_fixture):
  """test_audio_ops"""
  args, func, expected = fixture_lookup(io_data_fixture)

  entries = func(args)
  assert np.array_equal(entries, expected)

# A tf.data pipeline runs in graph mode and shape inference is invoked.
@pytest.mark.parametrize(
    ("io_data_fixture"),
    [
        pytest.param("resample"),
        pytest.param("decode_wav"),
        pytest.param("decode_flac"),
        pytest.param("encode_flac"),
    ],
    ids=[
        "resample",
        "decode_wav",
        "decode_flac",
        "encode_flac",
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
  assert np.array_equal(entries, expected)
