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
"""Test IOTensor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pytest

import tensorflow as tf
import tensorflow_io as tfio

def test_window():
  """test_window"""
  value = [[e] for e in range(100)]
  value = tfio.IOTensor.from_tensor(tf.constant(value))
  value = value.window(3)
  expected_value = [[e, e+1, e+2] for e in range(98)]
  assert np.all(value.to_tensor() == expected_value)

  v = tfio.IOTensor.from_tensor(tf.constant([1, 2, 3, 4, 5]))
  v = v.window(3)
  assert np.all(v.to_tensor() == [[1, 2, 3], [2, 3, 4], [3, 4, 5]])

@pytest.fixture(name="fixture_lookup")
def fixture_lookup_func(request):
  def _fixture_lookup(name):
    return request.getfixturevalue(name)
  return _fixture_lookup

@pytest.fixture(name="audio_wav", scope="module")
def fixture_audio_wav():
  """fixture_audio_wav"""
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "mono_10khz.wav")
  audio = tf.audio.decode_wav(tf.io.read_file(path))
  value = audio.audio * (1 << 15)
  value = tf.cast(value, tf.int16)

  args = path
  func = lambda e: tfio.IOTensor.graph(tf.int16).from_audio(e)
  expected = [v for _, v in enumerate(value)]
  meta_func = lambda e: e.rate
  meta_expected = 10000

  return args, func, expected, meta_func, meta_expected

@pytest.fixture(name="audio_wav_24", scope="module")
def fixture_audio_wav_24():
  """fixture_audio_wav_24"""
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "example_0.5s.wav")
  # raw was geenrated from:
  # $ sox example_0.5s.wav example_0.5s.s32
  raw_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "example_0.5s.s32")
  value = np.fromfile(raw_path, np.int32)
  value = np.reshape(value, [22050, 2])
  value = tf.constant(value)

  args = path
  func = lambda args: tfio.IOTensor.graph(tf.int32).from_audio(args)
  expected = [v for _, v in enumerate(value)]
  meta_func = lambda e: e.rate
  meta_expected = 44100

  return args, func, expected, meta_func, meta_expected

@pytest.fixture(name="audio_ogg", scope="module")
def fixture_audio_ogg():
  """fixture_audio_ogg"""
  # File is from the following with wav generated from `oggdec`.
  # https://en.wikipedia.org/wiki/File:Crescendo_example.ogg
  ogg_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "Crescendo_example.ogg")
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "Crescendo_example.wav")
  audio = tf.audio.decode_wav(tf.io.read_file(path))
  value = audio.audio * (1 << 15)
  value = tf.cast(value, tf.int16)

  args = ogg_path
  func = lambda args: tfio.IOTensor.graph(tf.int16).from_audio(args)
  expected = [v for _, v in enumerate(value)]
  meta_func = lambda e: e.rate
  meta_expected = 44100

  return args, func, expected, meta_func, meta_expected

@pytest.fixture(name="audio_flac", scope="module")
def fixture_audio_flac():
  """fixture_audio_flac"""
  # Sample from the following:
  # https://docs.espressif.com/projects/esp-adf/en/latest/design-guide/audio-samples.html
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "gs-16b-2c-44100hz.flac")
  wav_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "gs-16b-2c-44100hz.wav")
  audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
  value = audio.audio * (1 << 15)
  value = tf.cast(value, tf.int16)

  args = path
  func = lambda args: tfio.IOTensor.graph(tf.int16).from_audio(args)
  expected = [v for _, v in enumerate(value)]
  meta_func = lambda e: e.rate
  meta_expected = 44100

  return args, func, expected, meta_func, meta_expected

@pytest.mark.parametrize(
    ("io_tensor_fixture"),
    [
        pytest.param("audio_wav"),
        pytest.param("audio_wav_24"),
        pytest.param("audio_ogg"),
        pytest.param("audio_flac"),
    ],
    ids=[
        "audio[wav]",
        "audio[wav/24bit]",
        "audio[ogg]",
        "audio[flac]",
    ],
)
def test_io_tensor(fixture_lookup, io_tensor_fixture):
  """test_io_tensor"""
  args, func, expected, meta_func, meta_expected = fixture_lookup(
      io_tensor_fixture)

  io_tensor = func(args)
  io_tensor_meta = meta_func(io_tensor)

  # Test meta
  assert io_tensor_meta == meta_expected

  # Test to_tensor
  entries = io_tensor.to_tensor()
  assert len(entries) == len(expected)
  assert np.array_equal(entries, expected)

  # Test of io_tensor within dataset

  # Note: @tf.function is actually not needed, as tf.data.Dataset
  # will automatically wrap the `func` into a graph anyway.
  # The following is purely for explanation purposes.
  @tf.function
  def f(v):
    return func(v)[0:1000]

  args_dataset = tf.data.Dataset.from_tensor_slices([args])

  # Test with num_parallel_calls None, 1, 2
  for num_parallel_calls in [None, 1, 2]:
    dataset = args_dataset.map(f, num_parallel_calls=num_parallel_calls)

    item = 0
    # Notice dataset in dataset:
    for entries in dataset:
      assert len(entries) == len(expected[:1000])
      assert np.array_equal(entries, expected[:1000])
      item += 1
    assert item == 1

@pytest.mark.benchmark(
    group="io_tensor",
)
@pytest.mark.parametrize(
    ("io_tensor_fixture"),
    [
        pytest.param("audio_wav"),
        pytest.param("audio_wav_24"),
        pytest.param(
            "audio_ogg",
            marks=[
                pytest.mark.skip(reason="TODO(performance)"),
            ],
        ),
        pytest.param(
            "audio_flac",
            marks=[
                pytest.mark.skip(reason="TODO(performance)"),
            ],
        ),
    ],
    ids=[
        "audio[wav]",
        "audio[wav/24bit]",
        "audio[ogg]",
        "audio[flac]",
    ],
)
def test_io_tensor_benchmark(benchmark, fixture_lookup, io_tensor_fixture):
  """test_io_tensor_benchmark"""
  args, func, expected, _, _ = fixture_lookup(
      io_tensor_fixture)

  def f(v):
    io_tensor = func(v)
    return io_tensor.to_tensor()

  entries = benchmark(f, args)

  assert np.array_equal(entries, expected)
