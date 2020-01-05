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
"""Test Serialization"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

import tensorflow as tf
import tensorflow_io as tfio

@pytest.fixture(name="fixture_lookup")
def fixture_lookup_func(request):
  def _fixture_lookup(name):
    return request.getfixturevalue(name)
  return _fixture_lookup

@pytest.fixture(name="json", scope="module")
def fixture_json():
  """fixture_json"""
  data = """{
    "R": {
      "Foo": 208.82240295410156,
      "Bar": 93
    },
    "Background": [
      "~/0109.jpg",
      "~/0110.jpg"
    ],
    "Focal Length": 36.9439697265625,
    "Location": [
      7.8685874938964844,
      -4.7373886108398438,
      -0.038147926330566406],
    "Rotation": [
      -4.592592716217041,
      -4.4698805809020996,
      -6.9197754859924316]
   }
  """
  value = {
      "R": {
          "Foo": tf.constant(208.82240295410156, tf.float64),
          "Bar": tf.constant(93, tf.int64)
      },
      "Background": (
          tf.constant("~/0109.jpg", tf.string),
          tf.constant("~/0110.jpg", tf.string)
      ),
      "Location": tf.constant([
          7.8685874938964844,
          -4.7373886108398438,
          -0.038147926330566406
      ], tf.float64),
      "Rotation": tf.constant([
          -4.592592716217041,
          -4.4698805809020996,
          -6.9197754859924316
      ], tf.float64)
  }
  specs = {
      "R": {
          "Foo": tf.TensorSpec(tf.TensorShape([]), tf.float64),
          "Bar": tf.TensorSpec(tf.TensorShape([]), tf.int64)
      },
      "Background": (
          tf.TensorSpec(tf.TensorShape([]), tf.string),
          tf.TensorSpec(tf.TensorShape([]), tf.string)
      ),
      "Location": tf.TensorSpec(tf.TensorShape([3]), tf.float64),
      "Rotation": tf.TensorSpec(tf.TensorShape([3]), tf.float64)
  }

  return data, value, specs

@pytest.mark.parametrize(
    ("serialization_fixture", "decode_func", "encode_func"),
    [
        pytest.param(
            "json",
            tfio.experimental.serialization.decode_json,
            #tfio.experimental.serialization.encode_json),
            None),
    ],
    ids=[
        "json",
    ],
)
def test_serialization(
      fixture_lookup, serialization_fixture, decode_func, encode_func):
  """test_serialization_decode"""
  data, expected, specs = fixture_lookup(serialization_fixture)

  value = decode_func(data, specs)
  tf.nest.assert_same_structure(value, expected)
  assert all([
      np.array_equal(v, e) for v, e in zip(
          tf.nest.flatten(value), tf.nest.flatten(expected))])

@pytest.mark.parametrize(
    ("serialization_fixture", "decode_func", "encode_func"),
    [
        pytest.param(
            "json",
            tfio.experimental.serialization.decode_json,
            #tfio.experimental.serialization.encode_json),
            None),
    ],
    ids=[
        "json",
    ],
)
def test_serialization_in_dataset(
    fixture_lookup, serialization_fixture, decode_func, encode_func):
  """test_serialization_decode_in_dataset"""
  data, expected, specs = fixture_lookup(serialization_fixture)

  dataset = tf.data.Dataset.from_tensor_slices([data, data])
  dataset = dataset.map(lambda e: decode_func(e, specs))
  entries = list(dataset)

  assert len(entries) == 2
  for value in entries:
    tf.nest.assert_same_structure(value, expected)
    assert all([
        np.array_equal(v, e) for v, e in zip(
            tf.nest.flatten(value), tf.nest.flatten(expected))])
