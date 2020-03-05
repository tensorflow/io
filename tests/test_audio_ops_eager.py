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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

@pytest.mark.parametrize(
    ("io_data_fixture"),
    [
        pytest.param("resample"),
    ],
    ids=[
        "resample",
    ],
)
def test_audio_ops(fixture_lookup, io_data_fixture):
  """test_io_dataset_to_in_dataset"""
  args, func, expected = fixture_lookup(io_data_fixture)

  entries = func(args)
  assert np.array_equal(entries, expected)
