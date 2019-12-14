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
import shutil
import tempfile
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
      "test_audio", "ZASFX_ADSR_no_sustain.wav")
  audio = tf.audio.decode_wav(tf.io.read_file(path))
  value = audio.audio * (1 << 15)
  value = tf.cast(value, tf.int16)

  args = path
  func = lambda e: tfio.IOTensor.graph(tf.int16).from_audio(e)
  expected = value

  return args, func, expected

@pytest.fixture(name="audio_rate_wav", scope="module")
def fixture_audio_rate_wav():
  """fixture_audio_rate_wav"""
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.wav")

  args = path
  func = lambda e: tfio.IOTensor.graph(tf.int16).from_audio(e).rate
  expected = tf.constant(44100)

  return args, func, expected

@pytest.fixture(name="audio_wav_24", scope="module")
def fixture_audio_wav_24():
  """fixture_audio_wav_24"""
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.24.wav")
  raw_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.24.s32")
  value = np.fromfile(raw_path, np.int32)
  value = np.reshape(value, [14336, 2])
  value = tf.constant(value)

  args = path
  func = lambda args: tfio.IOTensor.graph(tf.int32).from_audio(args)
  expected = value

  return args, func, expected

@pytest.fixture(name="audio_rate_wav_24", scope="module")
def fixture_audio_rate_wav_24():
  """fixture_audio_rate_wav_24"""
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.24.wav")

  args = path
  func = lambda args: tfio.IOTensor.graph(tf.int32).from_audio(args).rate
  expected = tf.constant(44100)

  return args, func, expected

@pytest.fixture(name="audio_ogg", scope="module")
def fixture_audio_ogg():
  """fixture_audio_ogg"""
  ogg_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.ogg")
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.wav")
  audio = tf.audio.decode_wav(tf.io.read_file(path))
  value = audio.audio * (1 << 15)
  value = tf.cast(value, tf.int16)

  args = ogg_path
  func = lambda args: tfio.IOTensor.graph(tf.int16).from_audio(args)
  expected = value

  return args, func, expected

@pytest.fixture(name="audio_rate_ogg", scope="module")
def fixture_audio_rate_ogg():
  """fixture_audio_rate_ogg"""
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.ogg")

  args = path
  func = lambda args: tfio.IOTensor.graph(tf.int16).from_audio(args).rate
  expected = tf.constant(44100)

  return args, func, expected

@pytest.fixture(name="audio_flac", scope="module")
def fixture_audio_flac():
  """fixture_audio_flac"""
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.flac")
  wav_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.wav")
  audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
  value = audio.audio * (1 << 15)
  value = tf.cast(value, tf.int16)

  args = path
  func = lambda args: tfio.IOTensor.graph(tf.int16).from_audio(args)
  expected = value

  return args, func, expected

@pytest.fixture(name="audio_rate_flac", scope="module")
def fixture_audio_rate_flac():
  """fixture_audio_rate_flac"""
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "ZASFX_ADSR_no_sustain.flac")

  args = path
  func = lambda args: tfio.IOTensor.graph(tf.int16).from_audio(args).rate
  expected = tf.constant(44100)

  return args, func, expected

@pytest.fixture(name="hdf5", scope="module")
def fixture_hdf5(request):
  """fixture_hdf5"""
  import h5py # pylint: disable=import-outside-toplevel

  tmp_path = tempfile.mkdtemp()
  filename = os.path.join(tmp_path, "test.h5")

  data = list(range(5000))

  with h5py.File(filename, 'w') as f:
    f.create_dataset('float64', data=np.asarray(data, np.float64), dtype='f8')
  args = filename
  def func(args):
    return tfio.IOTensor.from_hdf5(args)('/float64')
  expected = np.asarray(data, np.float64).tolist()

  def fin():
    shutil.rmtree(tmp_path)
  request.addfinalizer(fin)

  return args, func, expected

# slice (__getitem__) is the most basic operation for IOTensor
@pytest.mark.parametrize(
    ("io_tensor_fixture"),
    [
        pytest.param("audio_wav"),
        pytest.param("audio_wav_24"),
        pytest.param("audio_ogg"),
        pytest.param("audio_flac"),
        pytest.param("hdf5"),
    ],
    ids=[
        "audio[wav]",
        "audio[wav/24bit]",
        "audio[ogg]",
        "audio[flac]",
        "hdf5",
    ],
)
def test_io_tensor_slice(fixture_lookup, io_tensor_fixture):
  """test_io_tensor_slice"""
  args, func, expected = fixture_lookup(io_tensor_fixture)

  io_tensor = func(args)

  # Test to_tensor
  entries = io_tensor.to_tensor()
  assert len(entries) == len(expected)
  assert np.array_equal(entries, expected)

  # Test __getitem__, use 7 to partition
  indices = list(range(0, len(expected), 7))
  for start, stop in list(zip(indices, indices[1:] + [len(expected)])):
    assert np.array_equal(io_tensor[start:stop], expected[start:stop])

# slice (__getitem__) could also be inside dataset for GraphIOTensor
@pytest.mark.parametrize(
    ("io_tensor_fixture", "num_parallel_calls"),
    [
        pytest.param("audio_wav", None),
        pytest.param("audio_wav", 2),
        pytest.param("audio_wav_24", None),
        pytest.param("audio_wav_24", 2),
        pytest.param("audio_ogg", None),
        pytest.param("audio_ogg", 2),
        pytest.param("audio_flac", None),
        pytest.param("audio_flac", 2),
        pytest.param(
            "hdf5", None,
            marks=[
                pytest.mark.skip(reason="TODO"),
            ],
        ),
        pytest.param(
            "hdf5", 2,
            marks=[
                pytest.mark.skip(reason="TODO"),
            ],
        ),
    ],
    ids=[
        "audio[wav]",
        "audio[wav]|2",
        "audio[wav/24bit]",
        "audio[wav/24bit]|2",
        "audio[ogg]",
        "audio[ogg]|2",
        "audio[flac]",
        "audio[flac]|2",
        "hdf5",
        "hdf5|2",
    ],
)
def test_io_tensor_slice_in_dataset(
    fixture_lookup, io_tensor_fixture, num_parallel_calls):
  """test_io_tensor_slice_in_dataset"""
  args, func, expected = fixture_lookup(io_tensor_fixture)

  # Test to_tensor within dataset

  # Note: @tf.function is actually not needed, as tf.data.Dataset
  # will automatically wrap the `func` into a graph anyway.
  # The following is purely for explanation purposes.
  @tf.function
  def f(e):
    return func(e).to_tensor()

  dataset = tf.data.Dataset.from_tensor_slices([args, args])
  dataset = dataset.map(f, num_parallel_calls=num_parallel_calls)

  item = 0
  for entries in dataset:
    assert len(entries) == len(expected)
    assert np.array_equal(entries, expected)
    item += 1
  assert item == 2

  # Note: @tf.function is actually not needed, as tf.data.Dataset
  # will automatically wrap the `func` into a graph anyway.
  # The following is purely for explanation purposes.
  @tf.function
  def g(e):
    return func(e)[0:100]

  dataset = tf.data.Dataset.from_tensor_slices([args, args])
  dataset = dataset.map(g, num_parallel_calls=num_parallel_calls)

  item = 0
  for entries in dataset:
    assert len(entries) == len(expected[:100])
    assert np.array_equal(entries, expected[:100])
    item += 1
  assert item == 2

# meta is supported for IOTensor
@pytest.mark.parametrize(
    ("io_tensor_fixture"),
    [
        pytest.param("audio_rate_wav"),
        pytest.param("audio_rate_wav_24"),
        pytest.param("audio_rate_ogg"),
        pytest.param("audio_rate_flac"),
    ],
    ids=[
        "audio[rate][wav]",
        "audio[rate][wav/24bit]",
        "audio[rate][ogg]",
        "audio[rate][flac]",
    ],
)
def test_io_tensor_meta(fixture_lookup, io_tensor_fixture):
  """test_io_tensor_slice"""
  args, func, expected = fixture_lookup(io_tensor_fixture)

  # Test meta data attached to IOTensor
  meta = func(args)
  assert meta == expected

# meta inside dataset is also supported for GraphIOTensor
@pytest.mark.parametrize(
    ("io_tensor_fixture"),
    [
        pytest.param("audio_rate_wav"),
        pytest.param("audio_rate_wav_24"),
        pytest.param("audio_rate_ogg"),
        pytest.param("audio_rate_flac"),
    ],
    ids=[
        "audio[rate][wav]",
        "audio[rate][wav/24bit]",
        "audio[rate][ogg]",
        "audio[rate][flac]",
    ],
)
def test_io_tensor_meta_in_dataset(fixture_lookup, io_tensor_fixture):
  """test_io_tensor_slice"""
  args, func, expected = fixture_lookup(io_tensor_fixture)

  # Note: @tf.function is actually not needed, as tf.data.Dataset
  # will automatically wrap the `func` into a graph anyway.
  # The following is purely for explanation purposes.
  @tf.function
  def f(e):
    return func(e)

  dataset = tf.data.Dataset.from_tensor_slices([args, args])
  dataset = dataset.map(f)

  item = 0
  for meta in dataset:
    assert meta == expected
    item += 1
  assert item == 2

# This is the basic benchmark for IOTensor.
@pytest.mark.benchmark(
    group="io_tensor",
)
@pytest.mark.parametrize(
    ("io_tensor_fixture"),
    [
        pytest.param("audio_wav"),
        pytest.param("audio_wav_24"),
        pytest.param("audio_ogg"),
        pytest.param("audio_flac"),
        pytest.param("hdf5"),
    ],
    ids=[
        "audio[wav]",
        "audio[wav/24bit]",
        "audio[ogg]",
        "audio[flac]",
        "hdf5",
    ],
)
def test_io_tensor_benchmark(benchmark, fixture_lookup, io_tensor_fixture):
  """test_io_tensor_benchmark"""
  args, func, expected = fixture_lookup(io_tensor_fixture)

  def f(v):
    io_tensor = func(v)
    return io_tensor.to_tensor()

  entries = benchmark(f, args)

  assert np.array_equal(entries, expected)
