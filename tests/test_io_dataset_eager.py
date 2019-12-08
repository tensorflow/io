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
"""Test IODataset"""
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

@pytest.fixture(name="fixture_lookup")
def fixture_lookup_func(request):
  def _fixture_lookup(name):
    return request.getfixturevalue(name)
  return _fixture_lookup

@pytest.fixture(name="lmdb")
def fixture_lmdb(request):
  """fixture_lmdb"""
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_lmdb", "data.mdb")
  tmp_path = tempfile.mkdtemp()
  filename = os.path.join(tmp_path, "data.mdb")
  shutil.copy(path, filename)

  def fin():
    shutil.rmtree(tmp_path)
  request.addfinalizer(fin)

  args = filename
  func = tfio.IODataset.from_lmdb
  expected = [
      (str(i).encode(), str(chr(ord("a") + i)).encode()) for i in range(10)]

  return args, func, expected

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
  func = lambda args: tfio.IODataset.graph(tf.int16).from_audio(args)
  expected = [v for _, v in enumerate(value)]

  return args, func, expected

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
  func = lambda args: tfio.IODataset.graph(tf.int32).from_audio(args)
  expected = [v for _, v in enumerate(value)]

  return args, func, expected

@pytest.fixture(name="audio_ogg", scope="module")
def fixture_audio_ogg():
  """fixture_audio_ogg"""
  # File is from the following with wav generated from `oggdec`.
  # https://en.wikipedia.org/wiki/File:Crescendo_example.ogg
  # The length is too long so cut the first 10000 samples.
  ogg_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "Crescendo_example.ogg")
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "Crescendo_example.wav")
  audio = tf.audio.decode_wav(tf.io.read_file(path))
  value = audio.audio * (1 << 15)
  value = tf.cast(value[:10000], tf.int16)

  args = ogg_path
  func = lambda args: tfio.IODataset.graph(tf.int16).from_audio(args).take(10000)
  expected = [v for _, v in enumerate(value)]

  return args, func, expected

@pytest.fixture(name="audio_flac", scope="module")
def fixture_audio_flac():
  """fixture_audio_flac"""
  # Sample from the following:
  # https://docs.espressif.com/projects/esp-adf/en/latest/design-guide/audio-samples.html
  # The length is too long so cut the first 10000 samples.
  path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "gs-16b-2c-44100hz.flac")
  wav_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_audio", "gs-16b-2c-44100hz.wav")
  audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
  value = audio.audio * (1 << 15)
  value = tf.cast(value[:10000], tf.int16)

  args = path
  func = lambda args: tfio.IODataset.graph(tf.int16).from_audio(args).take(10000)
  expected = [v for _, v in enumerate(value)]

  return args, func, expected

@pytest.mark.parametrize(
    ("io_dataset_fixture"),
    [
        pytest.param(
            "lmdb",
            marks=[
                pytest.mark.xfail(reason="TODO"),
            ],
        ),
        pytest.param("audio_wav"),
        pytest.param("audio_wav_24"),
        pytest.param("audio_ogg"),
        pytest.param("audio_flac"),
    ],
    ids=[
        "lmdb",
        "audio[wav]",
        "audio[wav/24bit]",
        "audio[ogg]",
        "audio[flac]",
    ],
)
def test_io_dataset(fixture_lookup, io_dataset_fixture):
  """test_io_dataset"""
  args, func, expected = fixture_lookup(io_dataset_fixture)

  dataset = func(args)

  # Run of dataset iteration
  entries = [e for e in dataset]

  assert len(entries) == len(expected)
  assert all([np.array_equal(a, b) for (a, b) in zip(entries, expected)])

  # A re-run of dataset iteration will yield the same result,
  # this is needed for training with tf.keras
  entries = [e for e in dataset]

  assert len(entries) == len(expected)
  assert all([np.array_equal(a, b) for (a, b) in zip(entries, expected)])

  # Test of take
  expected_taken = expected[:5]
  entries_taken = [e for e in dataset.take(5)]

  assert len(entries_taken) == len(expected_taken)
  assert all([
      np.array_equal(a, b) for (a, b) in zip(entries_taken, expected_taken)])

  # Test of batch
  indices = list(range(0, len(entries), 3))
  indices = list(zip(indices, indices[1:] + [len(entries)]))
  expected_batched = [expected[i:j] for i, j in indices]

  entries_batched = [e for e in dataset.batch(3)]

  assert len(entries_batched) == len(expected_batched)
  assert all([
      all([np.array_equal(i, j) for (i, j) in zip(a, b)]) for (a, b) in zip(
          entries_batched, expected_batched)])

  # Test of dataset within dataset

  # Note: @tf.function is actually not needed, as tf.data.Dataset
  # will automatically wrap the `func` into a graph anyway.
  # The following is purely for explanation purposes.
  @tf.function
  def f(v):
    return func(v)

  args_dataset = tf.data.Dataset.from_tensor_slices([args])

  # Test with num_parallel_calls None, 1, 2
  for num_parallel_calls in [None, 1, 2]:
    dataset = args_dataset.map(f, num_parallel_calls=num_parallel_calls)

    item = 0
    # Notice dataset in dataset:
    for d in dataset:
      i = 0
      for v in d:
        assert np.array_equal(expected[i], v)
        i += 1
      assert i == len(expected)
      item += 1
    assert item == 1

@pytest.mark.benchmark(
    group="io_dataset",
)
@pytest.mark.parametrize(
    ("io_dataset_fixture"),
    [
        pytest.param("lmdb"),
        pytest.param("audio_wav"),
        pytest.param("audio_wav_24"),
        pytest.param("audio_ogg"),
        pytest.param("audio_flac"),
    ],
    ids=[
        "lmdb",
        "audio[wav]",
        "audio[wav/24bit]",
        "audio[ogg]",
        "audio[flac]",
    ],
)
def test_io_dataset_benchmark(benchmark, fixture_lookup, io_dataset_fixture):
  """test_io_dataset_benchmark"""
  args, func, expected = fixture_lookup(io_dataset_fixture)

  def f(v):
    dataset = func(v)
    entries = [e for e in dataset]
    return entries

  entries = benchmark(f, args)

  assert len(entries) == len(expected)
  assert all([np.array_equal(a, b) for (a, b) in zip(entries, expected)])
