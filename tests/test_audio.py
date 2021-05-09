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
"""Test Audio Dataset"""

import os
import sys
import pytest
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio


@pytest.fixture(name="audio_data", scope="module")
def fixture_audio_data():
    """fixture_audio_data"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_audio", "mono_10khz.wav"
    )
    audio = tf.audio.decode_wav(tf.io.read_file(path))
    value = audio.audio * (1 << 15)
    value = tf.cast(value, tf.int16)
    rate = audio.sample_rate
    return path, value, rate


@pytest.fixture(name="audio_data_24", scope="module")
def fixture_audio_data_24():
    """fixture_audio_data_24"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.24.wav",
    )
    raw_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.24.s32",
    )
    value = np.fromfile(raw_path, np.int32)
    value = np.reshape(value, [14336, 2])
    value = tf.constant(value)
    rate = tf.constant(44100)
    return path, value, rate


@pytest.mark.parametrize(
    ("io_tensor_func"),
    [
        pytest.param(
            lambda f: tfio.IOTensor.from_ffmpeg(f)("a:0"),
            marks=[
                pytest.mark.skipif(
                    sys.platform == "darwin", reason="TODO: !!!pytest-xdist!!!"
                ),
                pytest.mark.xfail(reason="does not support 24 bit yet"),
            ],
        ),
    ],
    ids=["from_ffmpeg"],
)
def test_audio_io_tensor_24(audio_data_24, io_tensor_func):
    """test_audio_io_tensor_24"""
    audio_path, audio_value, audio_rate = audio_data_24

    audio_tensor = io_tensor_func(audio_path)
    assert audio_tensor.rate == audio_rate
    assert audio_tensor.shape == audio_value.shape
    assert np.all(audio_tensor.to_tensor() == audio_value)
    for step in [1, 100, 101, 200, 501, 600, 1001, 2000, 5001]:
        indices = list(range(0, 14336, step))
        # TODO: -1 vs. 14336 might need fix
        for (start, stop) in zip(indices, indices[1:] + [14336]):
            audio_tensor_value = audio_tensor[start:stop]
            audio_value_value = audio_value[start:stop]
            assert audio_tensor_value.shape == audio_value_value.shape
            assert np.all(audio_tensor_value == audio_value_value)


@pytest.mark.parametrize(
    ("io_dataset_func"),
    [
        pytest.param(
            lambda f: tfio.IODataset.from_ffmpeg(f, "a:0"),
            marks=[
                pytest.mark.skipif(
                    sys.platform == "darwin", reason="TODO: !!!pytest-xdist!!!"
                ),
                pytest.mark.xfail(reason="does not support 24 bit yet"),
            ],
        ),
    ],
    ids=["from_ffmpeg"],
)
def test_audio_io_dataset_24(audio_data_24, io_dataset_func):
    """test_audio_io_dataset_24"""
    audio_path, audio_value, _ = audio_data_24

    audio_dataset = io_dataset_func(audio_path)

    i = 0
    for value in audio_dataset:
        assert value.shape == [2]
        assert np.all(audio_value[i] == value)
        i += 1
    assert i == 14336

    audio_dataset = io_dataset_func(audio_path).batch(2)

    i = 0
    for value in audio_dataset:
        assert value.shape == [2, 2]
        assert np.all(audio_value[i] == value[0])
        assert np.all(audio_value[i + 1] == value[1])
        i += 2
    assert i == 14336


@pytest.mark.parametrize(
    ("io_tensor_func"),
    [
        pytest.param(
            lambda f: tfio.IOTensor.from_ffmpeg(f)("a:0"),
            marks=[
                pytest.mark.skipif(
                    sys.platform == "darwin", reason="TODO: !!!pytest-xdist!!!"
                ),
                pytest.mark.xfail(reason="shape does not work correctly yet"),
            ],
        ),
    ],
    ids=["from_ffmpeg"],
)
def test_audio_io_tensor(audio_data, io_tensor_func):
    """test_audio_io_tensor"""
    audio_path, audio_value, audio_rate = audio_data

    audio_tensor = io_tensor_func(audio_path)
    assert audio_tensor.rate == audio_rate
    assert audio_tensor.shape == audio_value.shape
    assert np.all(audio_tensor.to_tensor() == audio_value)
    for step in [1, 100, 101, 200, 501, 600, 1001, 2000, 5001]:
        indices = list(range(0, 5760, step))
        # TODO: -1 vs. 5760 might need fix
        for (start, stop) in zip(indices, indices[1:] + [5760]):
            audio_tensor_value = audio_tensor[start:stop]
            audio_value_value = audio_value[start:stop]
            assert audio_tensor_value.shape == audio_value_value.shape
            assert np.all(audio_tensor_value == audio_value_value)


@pytest.mark.parametrize(
    ("io_dataset_func"),
    [
        pytest.param(
            lambda f: tfio.IODataset.graph(tf.int16).from_ffmpeg(f, "a:0"),
            marks=[
                pytest.mark.skipif(
                    sys.platform == "darwin", reason="TODO: !!!pytest-xdist!!!"
                ),
            ],
        ),
        pytest.param(
            lambda f: tfio.IODataset.from_ffmpeg(f, "a:0"),
            marks=[
                pytest.mark.skipif(
                    sys.platform == "darwin", reason="TODO: !!!pytest-xdist!!!"
                ),
            ],
        ),
    ],
    ids=["from_ffmpeg", "from_ffmpeg(eager)"],
)
def test_audio_io_dataset(audio_data, io_dataset_func):
    """test_audio_io_dataset"""
    audio_path, audio_value, _ = audio_data

    audio_dataset = io_dataset_func(audio_path)

    i = 0
    for value in audio_dataset:
        assert audio_value[i] == value
        i += 1
    assert i == 5760

    audio_dataset = io_dataset_func(audio_path).batch(2)

    i = 0
    for value in audio_dataset:
        assert audio_value[i] == value[0]
        assert audio_value[i + 1] == value[1]
        i += 2
    assert i == 5760


@pytest.mark.parametrize(
    ("io_tensor_func", "num_parallel_calls"),
    [
        pytest.param(
            tfio.IOTensor.from_ffmpeg,
            1,
            marks=[
                pytest.mark.skipif(
                    sys.platform == "darwin", reason="TODO: !!!pytest-xdist!!!"
                ),
                pytest.mark.xfail(reason="does not work in graph yet"),
            ],
        ),
    ],
    ids=["from_ffmpeg"],
)
def test_audio_io_tensor_with_dataset(audio_data, io_tensor_func, num_parallel_calls):
    """test_audio_io_dataset_with_dataset"""
    audio_path, audio_value, audio_rate = audio_data

    filename_dataset = tf.data.Dataset.from_tensor_slices([audio_path, audio_path])
    position_dataset = tf.data.Dataset.from_tensor_slices(
        [tf.constant(1000, tf.int64), tf.constant(2000, tf.int64)]
    )

    dataset = tf.data.Dataset.zip((filename_dataset, position_dataset))

    # Note: @tf.function is actually not needed, as tf.data.Dataset
    # will automatically wrap the `func` into a graph anyway.
    # The following is purely for explanation purposes.
    # Return: audio chunk from position:position+100, and the rate.
    @tf.function
    def func(filename, position):
        audio = io_tensor_func(filename)
        return audio[position : position + 100], audio.rate

    dataset = dataset.map(func, num_parallel_calls=num_parallel_calls)

    item = 0
    for (data, rate) in dataset:
        assert audio_rate == rate
        assert data.shape == (100, 1)
        position = 1000 if item == 0 else 2000
        for i in range(100):
            assert audio_value[position + i] == data[i]
        item += 1
    assert item == 2


@pytest.mark.parametrize(
    ("io_dataset_func"),
    [
        pytest.param(
            lambda f: tfio.IODataset.graph(tf.int16).from_ffmpeg(f, "a:0"),
            marks=[
                pytest.mark.skipif(
                    sys.platform == "darwin", reason="TODO: !!!pytest-xdist!!!"
                ),
            ],
        ),
    ],
    ids=["from_ffmpeg"],
)
def test_audio_io_dataset_with_dataset(audio_data, io_dataset_func):
    """test_audio_io_dataset_with_dataset"""
    audio_path, audio_value, _ = audio_data

    filename_dataset = tf.data.Dataset.from_tensor_slices([audio_path, audio_path])
    position_dataset = tf.data.Dataset.from_tensor_slices(
        [tf.constant(1000, tf.int64), tf.constant(2000, tf.int64)]
    )

    dataset = tf.data.Dataset.zip((filename_dataset, position_dataset))

    # Note: @tf.function is actually not needed, as tf.data.Dataset
    # will automatically wrap the `func` into a graph anyway.
    # The following is purely for explanation purposes.
    # Return: an embedded dataset (in an outer dataset) for position:position+100
    @tf.function
    def func(filename, position):
        audio_dataset = io_dataset_func(filename)
        return audio_dataset.skip(position).take(100)

    dataset = dataset.map(func)

    item = 0
    # Notice audio_dataset in dataset:
    for audio_dataset in dataset:
        position = 1000 if item == 0 else 2000
        i = 0
        for value in audio_dataset:
            assert audio_value[position + i] == value
            i += 1
        assert i == 100
        item += 1
    assert item == 2
