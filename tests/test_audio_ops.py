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


def fixture_resample_base():
    """fixture_resample_base"""
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
        "ZASFX_ADSR_no_sustain-4410-quality-default.wav",
    )
    expected_audio = tf.audio.decode_wav(tf.io.read_file(expected_path))
    expected_value = expected_audio.audio * (1 << 15)
    expected_value = tf.cast(expected_value, tf.int16)

    args = value
    func = lambda e: tfio.audio.resample(e, 44100, 4410)
    expected = expected_value

    return args, func, expected


@pytest.fixture(name="resample", scope="module")
def fixture_resample():
    """fixture_resample"""
    args, func, expected = fixture_resample_base()
    return args, func, expected


@pytest.fixture(name="resample_1d", scope="module")
def fixture_resample_1d():
    """fixture_resample_1d"""
    args, func, expected = fixture_resample_base()
    return args[:, 0], func, expected[:, 0]


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
    func = lambda e: tfio.audio.decode_wav(e, dtype=tf.int16)
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
    func = lambda e: tfio.audio.encode_wav(e, rate=44100)
    expected = value

    return args, func, expected


@pytest.fixture(name="decode_wav_u8", scope="module")
def fixture_decode_wav_u8():
    """fixture_decode_wav_u8"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.u8.wav",
    )
    content = tf.io.read_file(path)

    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio
    value = (value + 1.0) * 128.0
    value = tf.cast(value, tf.uint8)

    args = content

    def func(e):
        delta = tf.constant(3.0, tf.float32)
        v = tfio.audio.decode_wav(e, dtype=tf.uint8)
        v = tf.cast(v, tf.float32) - tf.cast(value, tf.float32)
        v = tf.math.logical_and(tf.math.less(v, delta), tf.math.greater(v, -delta))
        return v

    expected = tf.ones([14336, 2], tf.bool)

    return args, func, expected


@pytest.fixture(name="encode_wav_u8", scope="module")
def fixture_encode_wav_u8():
    """fixture_encode_wav_u8"""
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio
    value = (value + 1.0) / 2.0 * 256.0
    value = tf.cast(value, tf.uint8)

    args = value

    def func(e):
        v = tfio.audio.encode_wav(e, rate=44100)
        return tfio.audio.decode_wav(v, dtype=tf.uint8)

    expected = value

    return args, func, expected


@pytest.fixture(name="decode_wav_s24", scope="module")
def fixture_decode_wav_s24():
    """fixture_decode_wav_s24"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.s24.wav",
    )
    content = tf.io.read_file(path)

    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio * (1 << 31)
    value = tf.cast(value, tf.int32)

    args = content
    func = lambda e: tfio.audio.decode_wav(e, dtype=tf.int32)
    expected = value

    return args, func, expected


@pytest.fixture(name="encode_wav_s24", scope="module")
def fixture_encode_wav_s24():
    """fixture_encode_wav_s24"""
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio * (1 << 31)
    value = tf.cast(value, tf.int32)

    args = value

    def func(e):
        v = tfio.audio.encode_wav(e, rate=44100)
        return tfio.audio.decode_wav(v, dtype=tf.int32)

    expected = value

    return args, func, expected


@pytest.fixture(name="decode_wav_f32", scope="module")
def fixture_decode_wav_f32():
    """fixture_decode_wav_f32"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.f32.wav",
    )
    content = tf.io.read_file(path)

    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio

    args = content
    func = lambda e: tfio.audio.decode_wav(e, dtype=tf.float32)
    expected = value

    return args, func, expected


@pytest.fixture(name="encode_wav_f32", scope="module")
def fixture_encode_wav_f32():
    """fixture_encode_wav_f32"""
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio

    args = value

    def func(e):
        v = tfio.audio.encode_wav(e, rate=44100)
        return tfio.audio.decode_wav(v, dtype=tf.float32)

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
    func = lambda e: tfio.audio.decode_flac(e, dtype=tf.int16)
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
        v = tfio.audio.encode_flac(e, rate=44100)
        return tfio.audio.decode_flac(v, dtype=tf.int16)

    expected = value

    return args, func, expected


@pytest.fixture(name="decode_flac_u8", scope="module")
def fixture_decode_flac_u8():
    """fixture_decode_flac_u8"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.u8.flac",
    )
    content = tf.io.read_file(path)

    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio
    value = (value + 1.0) * 128.0
    value = tf.cast(value, tf.uint8)

    args = content

    def func(e):
        delta = tf.constant(3.0, tf.float32)
        v = tfio.audio.decode_flac(e, dtype=tf.uint8)
        v = tf.cast(v, tf.float32) - tf.cast(value, tf.float32)
        v = tf.math.logical_and(tf.math.less(v, delta), tf.math.greater(v, -delta))
        return v

    expected = tf.ones([14336, 2], tf.bool)

    return args, func, expected


@pytest.fixture(name="encode_flac_u8", scope="module")
def fixture_encode_flac_u8():
    """fixture_encode_flac_u8"""
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio
    value = (value + 1.0) / 2.0 * 256.0
    value = tf.cast(value, tf.uint8)

    args = value

    def func(e):
        v = tfio.audio.encode_flac(e, rate=44100)
        return tfio.audio.decode_flac(v, dtype=tf.uint8)

    expected = value

    return args, func, expected


@pytest.fixture(name="decode_flac_s24", scope="module")
def fixture_decode_flac_s24():
    """fixture_decode_flac_s24"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.s24.flac",
    )
    content = tf.io.read_file(path)

    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio

    args = content

    def func(e):
        v = tfio.audio.decode_flac(e, dtype=tf.int32)
        v = tf.cast(v, tf.float32) / tf.constant((1 << 31), tf.float32)
        return v

    expected = value

    return args, func, expected


@pytest.fixture(name="encode_flac_s24", scope="module")
def fixture_encode_flac_s24():
    """fixture_encode_flac_s24"""
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio * (1 << 31)
    value = tf.cast(value, tf.int32)

    args = value

    def func(e):
        v = tfio.audio.encode_flac(e, rate=44100)
        return tfio.audio.decode_flac(v, dtype=tf.int32)

    expected = value

    return args, func, expected


@pytest.fixture(name="decode_vorbis", scope="module")
def fixture_decode_vorbis():
    """fixture_decode_vorbis"""
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
        v = tfio.audio.decode_vorbis(e)
        v = v - value
        v = tf.math.logical_and(tf.math.less(v, delta), tf.math.greater(v, -delta))
        return v

    expected = tf.ones([14336, 2], tf.bool)

    return args, func, expected


@pytest.fixture(name="encode_vorbis", scope="module")
def fixture_encode_vorbis():
    """fixture_encode_vorbis"""
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
        v = tfio.audio.encode_vorbis(e, rate=44100)
        v = tfio.audio.decode_vorbis(v)
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
        v = tfio.audio.decode_mp3(e)
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
        v = tfio.audio.encode_mp3(e, rate=44100)
        v = tfio.audio.decode_mp3(v)
        v = tf.shape(v)
        return v

    # Should be [18816, 2] but lame expand additional samples
    expected = tf.constant([21888, 2], tf.int32)

    return args, func, expected


@pytest.fixture(name="decode_aac", scope="module")
def fixture_decode_aac():
    """fixture_decode_aac"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "gs-16b-2c-44100hz.mp4",
    )
    content = tf.io.read_file(path)

    # The test file gs-16b-2c-44100hz.wav is generated from the
    # method itself on macOS so it is not exactly a good test.
    # However, the manual playing of the file has been largely
    # match so we consider it ok.
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "gs-16b-2c-44100hz.ffmpeg.wav"
        if sys.platform == "linux"
        else "gs-16b-2c-44100hz.wav",
    )
    value = tfio.audio.decode_wav(tf.io.read_file(wav_path), dtype=tf.int16)

    args = content

    def func(e):
        delta = tf.constant(2.0, tf.float32)
        v = tfio.audio.decode_aac(e)
        v = tf.cast(v * (1 << 15), tf.int16)
        v = tf.cast(v, tf.float32) - tf.cast(value, tf.float32)
        v = tf.math.logical_and(tf.math.less(v, delta), tf.math.greater(v, -delta))
        return v

    expected = tf.ones_like(value, tf.bool)

    return args, func, expected


@pytest.fixture(name="encode_aac", scope="module")
def fixture_encode_aac():
    """fixture_encode_aac"""
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "gs-16b-2c-44100hz.wav",
    )
    value = tfio.audio.decode_wav(tf.io.read_file(wav_path), dtype=tf.int16)
    value = tf.cast(value, tf.float32) / 32768.0

    encoded_wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "gs-16b-2c-44100hz.encoded.ffmpeg.wav"
        if sys.platform == "linux"
        else "gs-16b-2c-44100hz.encoded.wav",
    )
    encoded_value = tfio.audio.decode_wav(
        tf.io.read_file(encoded_wav_path), dtype=tf.int16
    )

    # calculate the delta and expect a small diff
    args = value

    def func(e):
        delta = tf.constant(2.0, tf.float32)
        v = tfio.audio.encode_aac(e, rate=44100)
        v = tfio.audio.decode_aac(v)
        v = tf.cast(v * (1 << 15), tf.int16)
        v = tf.cast(v, tf.float32) - tf.cast(encoded_value, tf.float32)
        v = tf.math.logical_and(tf.math.less(v, delta), tf.math.greater(v, -delta))
        return v

    expected = tf.ones_like(encoded_value, tf.bool)

    return args, func, expected


# By default, operations runs in eager mode,
# Note as of now shape inference is skipped in eager mode
@pytest.mark.parametrize(
    ("io_data_fixture"),
    [
        pytest.param("resample"),
        pytest.param("resample_1d"),
        pytest.param("decode_wav"),
        pytest.param("encode_wav"),
        pytest.param("decode_wav_u8"),
        pytest.param("encode_wav_u8"),
        pytest.param("decode_wav_s24"),
        pytest.param("encode_wav_s24"),
        pytest.param("decode_wav_f32"),
        pytest.param("encode_wav_f32"),
        pytest.param("decode_flac"),
        pytest.param("encode_flac"),
        pytest.param("decode_flac_u8"),
        pytest.param("encode_flac_u8"),
        pytest.param("decode_flac_s24"),
        pytest.param("encode_flac_s24"),
        pytest.param("decode_vorbis"),
        pytest.param("encode_vorbis"),
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
        pytest.param(
            "decode_aac",
            marks=[
                pytest.mark.skipif(
                    (sys.platform == "linux" and sys.version_info < (3, 8))
                    or (sys.platform == "win32"),
                    reason="need ubuntu 20.04 which is python 3.8, and no windows",
                )
            ],
        ),
        pytest.param(
            "encode_aac",
            marks=[
                pytest.mark.skipif(
                    (sys.platform == "linux" and sys.version_info < (3, 8))
                    or (sys.platform == "win32"),
                    reason="need ubuntu 20.04 which is python 3.8, and no windows",
                )
            ],
        ),
    ],
    ids=[
        "resample",
        "resample[1d]",
        "decode_wav",
        "encode_wav",
        "decode_wav|u8",
        "encode_wav|u8",
        "decode_wav|s24",
        "encode_wav|s24",
        "decode_wav|f32",
        "encode_wav|f32",
        "decode_flac",
        "encode_flac",
        "decode_flac|u8",
        "encode_flac|u8",
        "decode_flac|s24",
        "encode_flac|s24",
        "decode_vorbis",
        "encode_vorbis",
        "decode_mp3",
        "encode_mp3",
        "decode_aac",
        "encode_aac",
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
        pytest.param("decode_wav_u8"),
        pytest.param("encode_wav_u8"),
        pytest.param("decode_wav_s24"),
        pytest.param("encode_wav_s24"),
        pytest.param("decode_wav_f32"),
        pytest.param("encode_wav_f32"),
        pytest.param("decode_flac"),
        pytest.param("encode_flac"),
        pytest.param("decode_flac_u8"),
        pytest.param("encode_flac_u8"),
        pytest.param("decode_flac_s24"),
        pytest.param("encode_flac_s24"),
        pytest.param("decode_vorbis"),
        pytest.param("encode_vorbis"),
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
        pytest.param(
            "decode_aac",
            marks=[
                pytest.mark.skipif(
                    (sys.platform == "linux" and sys.version_info < (3, 8))
                    or (sys.platform == "win32"),
                    reason=(
                        "need ubuntu 20.04 which is python 3.8, "
                        "and no windows support yet"
                    ),
                )
            ],
        ),
        pytest.param(
            "encode_aac",
            marks=[
                pytest.mark.skipif(
                    (sys.platform == "linux" and sys.version_info < (3, 8))
                    or (sys.platform == "win32"),
                    reason=(
                        "need ubuntu 20.04 which is python 3.8, "
                        "and no windows support yet"
                    ),
                )
            ],
        ),
    ],
    ids=[
        "resample",
        "decode_wav",
        "encode_wav",
        "decode_wav|u8",
        "encode_wav|u8",
        "decode_wav|s24",
        "encode_wav|s24",
        "decode_wav|f32",
        "encode_wav|f32",
        "decode_flac",
        "encode_flac",
        "decode_flac|u8",
        "encode_flac|u8",
        "decode_flac|s24",
        "encode_flac|s24",
        "decode_vorbis",
        "encode_vorbis",
        "decode_mp3",
        "encode_mp3",
        "decode_aac",
        "encode_aac",
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


@pytest.mark.skipif(
    sys.platform in ("win32", "darwin"), reason="no lame for darwin or win32",
)
def test_encode_mp3_mono():
    """test_encode_mp3_mono"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_audio", "mono_10khz.wav",
    )
    audio = tfio.audio.decode_wav(tf.io.read_file(path), dtype=tf.int16)
    assert audio.shape == [5760, 1]
    audio = tf.cast(audio, tf.float32) / 32768.0
    _ = tfio.audio.encode_mp3(audio, rate=8000)


def test_spectrogram():
    """test_spectrogram"""

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_audio", "mono_10khz.wav",
    )
    audio = tfio.audio.decode_wav(tf.io.read_file(path), dtype=tf.int16)
    assert audio.shape == [5760, 1]
    audio = tf.cast(audio, tf.float32) / 32768.0

    # TODO remove once spectrogram support channel > 1
    audio = tf.reshape(audio, [-1])

    nfft = 400
    window = 400
    stride = 200
    spectrogram = tfio.audio.spectrogram(audio, nfft=nfft, window=window, stride=stride)

    # TODO: assert content of spectrogram
    assert spectrogram.shape == [29, nfft // 2 + 1]
    assert spectrogram.dtype == tf.float32

    rate = 10000
    mels = 128
    fmin = 0
    fmax = 5000
    mel_spectrogram = tfio.audio.melscale(
        spectrogram, rate=rate, mels=mels, fmin=fmin, fmax=fmax
    )

    # TODO: assert content of mel_spectrogram
    assert mel_spectrogram.shape == [29, mels]
    assert mel_spectrogram.dtype == tf.float32

    dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)

    # TODO: assert content of dbscale_mel_spectrogram
    assert dbscale_mel_spectrogram.shape == [29, mels]
    assert dbscale_mel_spectrogram.dtype == tf.float32

    spec = dbscale_mel_spectrogram

    # Freq masking
    spec = tfio.audio.freq_mask(spec, param=30)

    # Time masking
    spec = tfio.audio.time_mask(spec, param=10)


def test_fade():
    """test_fade"""

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_audio", "mono_10khz.wav",
    )
    audio = tfio.audio.decode_wav(tf.io.read_file(path), dtype=tf.int16)
    assert audio.shape == [5760, 1]
    audio = tf.cast(audio, tf.float32) / 32768.0

    audio = audio[2000:4000]

    fade_in = 1000
    fade_out = 1500
    for mode in ["linear", "logarithmic", "exponential"]:
        fade = tfio.audio.fade(audio, fade_in=fade_in, fade_out=fade_out, mode=mode)

        # TODO: validate fade effect
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(audio.numpy())
        # plt.savefig("{}_data.png".format(mode))
        # plt.figure()
        # plt.plot(fade.numpy())
        # plt.savefig("{}_fade.png".format(mode))


def test_audio_trim_split_remix():
    """test_audio_trim_split_remix"""

    l = tf.range(0.0, 20.0, 1.0)
    r = tf.reverse(l, [0])
    v = tf.concat([l, r], axis=0)
    v = tf.broadcast_to(v, [3, 40])

    v0 = tf.reshape(v, [-1])
    v1 = tf.concat([v0[30:], tf.zeros([30], tf.float32)], axis=0)
    v2 = tf.concat([tf.zeros([60], tf.float32), v0[60:]], axis=0)
    v3 = tf.concat([tf.zeros([90], tf.float32), v0[90:]], axis=0)

    v = tf.stack([v0, v1, v2, v3])
    e = tf.constant([[5, 115], [0, 85], [60, 115], [90, 115]], tf.int64)
    i = tf.constant(
        [
            [[5, 35], [45, 75], [85, 115]],
            [[0, 5], [15, 45], [55, 85]],
            [[60, 75], [85, 115], [120, 120]],
            [[90, 115], [120, 120], [120, 120]],
        ],
        tf.int64,
    )

    epsilon = 4.95

    k0 = tf.boolean_mask(v0, tf.math.greater(v0, epsilon))
    k1 = tf.concat(
        [tf.boolean_mask(v1, tf.math.greater(v1, epsilon)), tf.zeros([25], tf.float32)],
        axis=0,
    )
    k2 = tf.concat(
        [tf.boolean_mask(v2, tf.math.greater(v2, epsilon)), tf.zeros([45], tf.float32)],
        axis=0,
    )
    k3 = tf.concat(
        [tf.boolean_mask(v3, tf.math.greater(v3, epsilon)), tf.zeros([65], tf.float32)],
        axis=0,
    )

    k = tf.stack([k0, k1, k2, k3])

    for value, axis, ee, ei, ek in [
        (v, 1, e, i, k),
        (
            tf.reshape(v, [4, 120, 1]),
            1,
            tf.reshape(e, [4, 2, 1]),
            tf.reshape(i, [4, 3, 2, 1]),
            tf.reshape(k, [4, 90, 1]),
        ),
    ]:
        indices = tfio.audio.trim(value, axis=axis, epsilon=epsilon)
        assert np.array_equal(ee, indices)

        indices = tfio.audio.split(value, axis=axis, epsilon=epsilon)
        assert np.array_equal(ei, indices)

        result = tfio.audio.remix(value, axis=axis, indices=indices)
        assert np.array_equal(ek, result)
