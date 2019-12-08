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

# Source of audio are based on the following:
#   https://commons.wikimedia.org/wiki/File:ZASFX_ADSR_no_sustain.ogg
# OGG: ZASFX_ADSR_no_sustain.ogg.
# WAV: oggdec ZASFX_ADSR_no_sustain.ogg # => ZASFX_ADSR_no_sustain.wav
# WAV (24 bit):
#   sox ZASFX_ADSR_no_sustain.wav -b 24 ZASFX_ADSR_no_sustain.24.wav
#   sox ZASFX_ADSR_no_sustain.24.wav ZASFX_ADSR_no_sustain.24.s32
# FLAC: ffmpeg -i ZASFX_ADSR_no_sustain.wav ZASFX_ADSR_no_sustain.flac
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
  func = lambda args: tfio.IODataset.graph(tf.int16).from_audio(args)
  expected = [v for _, v in enumerate(value)]

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
  func = lambda args: tfio.IODataset.graph(tf.int32).from_audio(args)
  expected = [v for _, v in enumerate(value)]

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
  func = lambda args: tfio.IODataset.graph(tf.int16).from_audio(args)
  expected = [v for _, v in enumerate(value)]

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
  func = lambda args: tfio.IODataset.graph(tf.int16).from_audio(args)
  expected = [v for _, v in enumerate(value)]

  return args, func, expected

# This test make sure dataset works in tf.keras inference.
# The requirement for tf.keras inference is the support of `iter()`:
#   entries = [e for e in dataset]
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
def test_io_dataset_basic(fixture_lookup, io_dataset_fixture):
  """test_io_dataset_basic"""
  args, func, expected = fixture_lookup(io_dataset_fixture)

  dataset = func(args)
  entries = [e for e in dataset]

  assert len(entries) == len(expected)
  assert all([np.array_equal(a, b) for (a, b) in zip(entries, expected)])

# This test makes sure basic dataset operations (take, batch) work.
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
def test_io_dataset_basic_operation(fixture_lookup, io_dataset_fixture):
  """test_io_dataset_basic_operation"""
  args, func, expected = fixture_lookup(io_dataset_fixture)

  dataset = func(args)

  # Test of take
  expected_taken = expected[:5]
  entries_taken = [e for e in dataset.take(5)]

  assert len(entries_taken) == len(expected_taken)
  assert all([
      np.array_equal(a, b) for (a, b) in zip(entries_taken, expected_taken)])

  # Test of batch
  indices = list(range(0, len(expected), 3))
  indices = list(zip(indices, indices[1:] + [len(expected)]))
  expected_batched = [expected[i:j] for i, j in indices]

  entries_batched = [e for e in dataset.batch(3)]

  assert len(entries_batched) == len(expected_batched)
  assert all([
      all([np.array_equal(i, j) for (i, j) in zip(a, b)]) for (a, b) in zip(
          entries_batched, expected_batched)])

# This test makes sure dataset works in tf.keras training.
# The requirement for tf.keras training is the support of multiple `iter()`
# runs with consistent result:
#   entries_1 = [e for e in dataset]
#   entries_2 = [e for e in dataset]
#   assert entries_1 = entries_2
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
def test_io_dataset_for_training(fixture_lookup, io_dataset_fixture):
  """test_io_dataset_for_training"""
  args, func, expected = fixture_lookup(io_dataset_fixture)

  dataset = func(args)

  # Run of dataset iteration
  entries = [e for e in dataset]

  assert len(entries) == len(expected)
  assert all([np.array_equal(a, b) for (a, b) in zip(entries, expected)])

  # A re-run of dataset iteration yield the same results, needed for training.
  entries = [e for e in dataset]

  assert len(entries) == len(expected)
  assert all([np.array_equal(a, b) for (a, b) in zip(entries, expected)])

# This test makes sure dataset in dataet and parallelism work.
# It is not needed for tf.keras but could be useful
# for complex data processing.
@pytest.mark.parametrize(
    ("io_dataset_fixture", "num_parallel_calls"),
    [
        pytest.param(
            "lmdb", None,
            marks=[
                pytest.mark.skip(reason="TODO"),
            ],
        ),
        pytest.param(
            "lmdb", 2,
            marks=[
                pytest.mark.skip(reason="TODO"),
            ],
        ),
        pytest.param("audio_wav", None),
        pytest.param("audio_wav", 2),
        pytest.param("audio_wav_24", None),
        pytest.param("audio_wav_24", 2),
        pytest.param("audio_ogg", None),
        pytest.param("audio_ogg", 2),
        pytest.param("audio_flac", None),
        pytest.param("audio_flac", 2),
    ],
    ids=[
        "lmdb",
        "lmdb|2",
        "audio[wav]",
        "audio[wav]|2",
        "audio[wav/24bit]",
        "audio[wav/24bit]|2",
        "audio[ogg]",
        "audio[ogg]|2",
        "audio[flac]",
        "audio[flac]|2",
    ],
)
def test_io_dataset_in_dataset_parallel(
    fixture_lookup, io_dataset_fixture, num_parallel_calls):
  """test_io_dataset_in_dataset_parallel"""
  args, func, expected = fixture_lookup(io_dataset_fixture)

  dataset = func(args)

  # Note: @tf.function is actually not needed, as tf.data.Dataset
  # will automatically wrap the `func` into a graph anyway.
  # The following is purely for explanation purposes.
  @tf.function
  def f(v):
    return func(v)

  args_dataset = tf.data.Dataset.from_tensor_slices([args, args])

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
  assert item == 2

# This test is a benchmark for dataset, could invoke/skip/disalbe through:
#   --benchmark-only
#   --benchmark-skip
#   --benchmark-disable
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
