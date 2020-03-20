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
import sys
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
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(path))
    value = audio.audio * (1 << 15)
    value = tf.cast(value, tf.int16)

    expected_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain-4410-quality-1.wav",
    )
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
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    content = tf.io.read_file(path)

    audio = tf.audio.decode_wav(tf.io.read_file(path))
    value = audio.audio * (1 << 15)
    value = tf.cast(value, tf.int16)

    args = content
    func = lambda e: tfio.experimental.audio.decode_wav(e, dtype=tf.int16)
    expected = value

    return args, func, expected


@pytest.fixture(name="encode_wav", scope="module")
def fixture_encode_wav():
    """fixture_encode_wav"""
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    value = tf.io.read_file(wav_path)
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.s16le.pcm",
    )
    audio = np.fromfile(path, np.int16)
    audio = np.reshape(audio, [14336, 2])
    audio = tf.convert_to_tensor(audio)

    args = audio
    func = lambda e: tfio.experimental.audio.encode_wav(e, rate=44100)
    expected = value

    return args, func, expected


@pytest.fixture(name="decode_flac", scope="module")
def fixture_decode_flac():
    """fixture_decode_flac"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.flac",
    )
    content = tf.io.read_file(path)

    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio * (1 << 15)
    value = tf.cast(value, tf.int16)

    args = content
    func = lambda e: tfio.experimental.audio.decode_flac(e, dtype=tf.int16)
    expected = value

    return args, func, expected


@pytest.fixture(name="encode_flac", scope="module")
def fixture_encode_flac():
    """fixture_encode_flac"""
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio * (1 << 15)
    value = tf.cast(value, tf.int16)

    args = value

    def func(e):
        v = tfio.experimental.audio.encode_flac(e, rate=44100)
        return tfio.experimental.audio.decode_flac(v, dtype=tf.int16)

    expected = value

    return args, func, expected


@pytest.fixture(name="decode_ogg", scope="module")
def fixture_decode_ogg():
    """fixture_decode_ogg"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.ogg",
    )
    content = tf.io.read_file(path)

    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio * (1 << 15)
    value = tf.cast(value, tf.float32) / 32768.0

    # calculate the delta and expect a small diff
    args = content

    def func(e):
        delta = tf.constant(0.00002, tf.float32)
        v = tfio.experimental.audio.decode_ogg(e)
        v = v - value
        v = tf.math.logical_and(tf.math.less(v, delta), tf.math.greater(v, -delta))
        return v

    expected = tf.ones([14336, 2], tf.bool)

    return args, func, expected


@pytest.fixture(name="encode_ogg", scope="module")
def fixture_encode_ogg():
    """fixture_encode_ogg"""
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio * (1 << 15)
    value = tf.cast(value, tf.float32) / 32768.0

    # calculate the delta and expect a small diff
    args = value

    def func(e):
        delta = tf.constant(0.05, tf.float32)
        v = tfio.experimental.audio.encode_ogg(e, rate=44100)
        v = tfio.experimental.audio.decode_ogg(v)
        v = v - e
        v = tf.math.logical_and(tf.math.less(v, delta), tf.math.greater(v, -delta))
        return v

    expected = tf.ones([14336, 2], tf.bool)

    return args, func, expected


@pytest.fixture(name="decode_mp3", scope="module")
def fixture_decode_mp3():
    """fixture_decode_mp3"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_audio", "l1-fl6.bit"
    )
    content = tf.io.read_file(path)
    raw_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_audio", "l1-fl6.raw"
    )
    raw = np.fromfile(raw_path, np.int16)
    raw = raw.reshape([-1, 2])
    value = tf.cast(raw, tf.float32) / 32768.0

    # calculate the delta and expect a small diff
    args = content

    def func(e):
        delta = tf.constant(0.00005, tf.float32)
        v = tfio.experimental.audio.decode_mp3(e)
        v = v - value
        v = tf.math.logical_and(tf.math.less(v, delta), tf.math.greater(v, -delta))
        return v

    expected = tf.ones([18816, 2], tf.bool)

    return args, func, expected


@pytest.fixture(name="encode_mp3", scope="module")
def fixture_encode_mp3():
    """fixture_encode_mp3"""
    raw_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_audio", "l1-fl6.raw"
    )
    raw = np.fromfile(raw_path, np.int16)
    raw = raw.reshape([-1, 2])
    value = tf.cast(raw, tf.float32) / 32768.0

    # lame has a delay which will expand the number of samples.
    # for that this test simply check the number of samples
    args = value

    def func(e):
        v = tfio.experimental.audio.encode_mp3(e, rate=44100)
        v = tfio.experimental.audio.decode_mp3(v)
        v = tf.shape(v)
        return v

    # Should be [18816, 2] but lame expand additional samples
    expected = tf.constant([21888, 2], tf.int32)

    return args, func, expected


# By default, operations runs in eager mode,
# Note as of now shape inference is skipped in eager mode
@pytest.mark.parametrize(
    ("io_data_fixture"),
    [
        pytest.param("resample"),
        pytest.param("decode_wav"),
        pytest.param("encode_wav"),
        pytest.param("decode_flac"),
        pytest.param("encode_flac"),
        pytest.param("decode_ogg"),
        pytest.param("encode_ogg"),
        pytest.param("decode_mp3"),
        pytest.param(
            "encode_mp3",
            marks=[
                pytest.mark.skipif(
                    sys.platform in ("win32", "darwin"),
                    reason="no lame for darwin or win32",
                ),
            ],
        ),
    ],
    ids=[
        "resample",
        "decode_wav",
        "encode_wav",
        "decode_flac",
        "encode_flac",
        "decode_ogg",
        "encode_ogg",
        "decode_mp3",
        "encode_mp3",
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
        pytest.param("encode_wav"),
        pytest.param("decode_flac"),
        pytest.param("encode_flac"),
        pytest.param("decode_ogg"),
        pytest.param("encode_ogg"),
        pytest.param("decode_mp3"),
        pytest.param(
            "encode_mp3",
            marks=[
                pytest.mark.skipif(
                    sys.platform in ("win32", "darwin"),
                    reason="no lame for darwin or win32",
                ),
            ],
        ),
    ],
    ids=[
        "resample",
        "decode_wav",
        "encode_wav",
        "decode_flac",
        "encode_flac",
        "decode_ogg",
        "encode_ogg",
        "decode_mp3",
        "encode_mp3",
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
