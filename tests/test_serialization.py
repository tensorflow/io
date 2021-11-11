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

import os
import json
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
            -6.9197754859924316],
        "Valid": true,
        "Boundary": [[10, 20],[30, 40]]    
    }
    """
    value = {
        "R": {
            "Foo": tf.constant(208.82240295410156, tf.float64),
            "Bar": tf.constant(93, tf.int64),
        },
        "Background": (
            tf.constant("~/0109.jpg", tf.string),
            tf.constant("~/0110.jpg", tf.string),
        ),
        "Location": tf.constant(
            [7.8685874938964844, -4.7373886108398438, -0.038147926330566406], tf.float64
        ),
        "Rotation": tf.constant(
            [-4.592592716217041, -4.4698805809020996, -6.9197754859924316], tf.float64
        ),
        "Valid": tf.constant(True, tf.bool),
        "Boundary": tf.constant([[10, 20], [30, 40]], tf.int64),
    }
    specs = {
        "R": {
            "Foo": tf.TensorSpec(tf.TensorShape([]), tf.float64),
            "Bar": tf.TensorSpec(tf.TensorShape([]), tf.int64),
        },
        "Background": (
            tf.TensorSpec(tf.TensorShape([]), tf.string),
            tf.TensorSpec(tf.TensorShape([]), tf.string),
        ),
        "Location": tf.TensorSpec(tf.TensorShape([3]), tf.float64),
        "Rotation": tf.TensorSpec(tf.TensorShape([3]), tf.float64),
        "Valid": tf.TensorSpec(tf.TensorShape([]), tf.bool),
        "Boundary": tf.TensorSpec(tf.TensorShape([2, 2]), tf.int64),
    }

    return data, value, specs


@pytest.fixture(name="avro", scope="module")
def fixture_avro():
    """fixture_avro"""
    avro_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_avro", "weather.avro"
    )
    with open(avro_path, "rb") as f:
        data = f.read()
    value = {
        "station": tf.constant("011990-99999", tf.string),
        "time": tf.constant(-619524000000, tf.int64),
        "temp": tf.constant(0, tf.int32),
    }
    avsc_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_avro", "weather.avsc"
    )
    with open(avsc_path, "rb") as f:
        specs = f.read()

    return data, value, specs


@pytest.mark.parametrize(
    ("serialization_fixture", "decode_function"),
    [
        pytest.param("json", tfio.experimental.serialization.decode_json),
        pytest.param("avro", tfio.experimental.serialization.decode_avro),
    ],
    ids=[
        "json",
        "avro",
    ],
)
def test_serialization_decode(fixture_lookup, serialization_fixture, decode_function):
    """test_serialization_decode"""
    data, value, specs = fixture_lookup(serialization_fixture)

    returned = decode_function(data, specs)
    tf.nest.assert_same_structure(value, returned)
    assert all(
        [
            np.array_equal(v, r)
            for v, r in zip(tf.nest.flatten(value), tf.nest.flatten(returned))
        ]
    )


@pytest.mark.parametrize(
    ("serialization_fixture", "encode_function", "decode_function"),
    [
        pytest.param(
            "avro",
            tfio.experimental.serialization.encode_avro,
            tfio.experimental.serialization.decode_avro,
        ),
    ],
    ids=[
        "avro",
    ],
)
def test_serialization_encode(
    fixture_lookup, serialization_fixture, encode_function, decode_function
):
    """test_serialization_encode"""
    _, value, specs = fixture_lookup(serialization_fixture)

    returned = encode_function(value, specs)
    returned = decode_function(returned, specs)
    tf.nest.assert_same_structure(value, returned)
    assert all(
        [
            np.array_equal(v, r)
            for v, r in zip(tf.nest.flatten(value), tf.nest.flatten(returned))
        ]
    )


@pytest.mark.parametrize(
    ("serialization_fixture", "decode_function"),
    [
        pytest.param("json", tfio.experimental.serialization.decode_json),
        pytest.param("avro", tfio.experimental.serialization.decode_avro),
    ],
    ids=[
        "json",
        "avro",
    ],
)
def test_serialization_decode_in_dataset(
    fixture_lookup, serialization_fixture, decode_function
):
    """test_serialization_decode_in_dataset"""
    data, value, specs = fixture_lookup(serialization_fixture)

    dataset = tf.data.Dataset.from_tensor_slices([data, data])
    dataset = dataset.map(lambda e: decode_function(e, specs))
    entries = list(dataset)

    assert len(entries) == 2
    for returned in entries:
        tf.nest.assert_same_structure(value, returned)
        assert all(
            [
                np.array_equal(v, r)
                for v, r in zip(tf.nest.flatten(value), tf.nest.flatten(returned))
            ]
        )


def test_decode_json_partial_shape():
    """Test case for partial shape GitHub 918."""
    r = json.dumps({"foo": [1, 2, 3, 4, 5]})

    @tf.function(autograph=False)
    def parse_json(json_text):
        specs = {"foo": tf.TensorSpec(tf.TensorShape([None]), tf.int32)}
        parsed = tfio.experimental.serialization.decode_json(json_text, specs)
        return parsed["foo"]

    v = parse_json(r)
    assert np.array_equal(v, [1, 2, 3, 4, 5])
