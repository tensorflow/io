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
    expected_value = [[e, e + 1, e + 2] for e in range(98)]
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
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(path))
    value = audio.audio * (1 << 15)
    value = tf.cast(value, tf.int16)

    args = path
    func = lambda e: tfio.IOTensor.graph(tf.int16).from_audio(e)
    expected = value

    return args, func, expected, np.array_equal


@pytest.fixture(name="audio_rate_wav", scope="module")
def fixture_audio_rate_wav():
    """fixture_audio_rate_wav"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )

    args = path
    func = lambda e: tfio.IOTensor.graph(tf.int16).from_audio(e).rate
    expected = tf.constant(44100)

    return args, func, expected, np.array_equal


@pytest.fixture(name="audio_wav_s24", scope="module")
def fixture_audio_wav_s24():
    """fixture_audio_wav_s24"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.s24.wav",
    )

    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio * (1 << 31)
    value = tf.cast(value, tf.int32)

    args = path
    func = lambda e: tfio.IOTensor.graph(tf.int32).from_audio(e)
    expected = value

    return args, func, expected, np.array_equal


@pytest.fixture(name="audio_rate_wav_s24", scope="module")
def fixture_audio_rate_wav_s24():
    """fixture_audio_rate_wav_s24"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.s24.wav",
    )

    args = path
    func = lambda e: tfio.IOTensor.graph(tf.int32).from_audio(e).rate
    expected = tf.constant(44100)

    return args, func, expected, np.array_equal


@pytest.fixture(name="audio_vorbis", scope="module")
def fixture_audio_vorbis():
    """fixture_audio_vorbis"""
    ogg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.ogg",
    )
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(path))
    value = audio.audio * (1 << 15)
    value = tf.cast(value, tf.float32) / 32768.0

    args = ogg_path
    func = lambda args: tfio.IOTensor.graph(tf.float32).from_audio(args)
    expected = value
    equal = lambda a, b: np.allclose(a, b, atol=0.00002)

    return args, func, expected, equal


@pytest.fixture(name="audio_rate_vorbis", scope="module")
def fixture_audio_rate_vorbis():
    """fixture_audio_rate_vorbis"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.ogg",
    )

    args = path
    func = lambda args: tfio.IOTensor.graph(tf.int16).from_audio(args).rate
    expected = tf.constant(44100)

    return args, func, expected, np.array_equal


@pytest.fixture(name="audio_flac", scope="module")
def fixture_audio_flac():
    """fixture_audio_flac"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.flac",
    )
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.wav",
    )
    audio = tf.audio.decode_wav(tf.io.read_file(wav_path))
    value = audio.audio * (1 << 15)
    value = tf.cast(value, tf.int16)

    args = path
    func = lambda args: tfio.IOTensor.graph(tf.int16).from_audio(args)
    expected = value

    return args, func, expected, np.array_equal


@pytest.fixture(name="audio_rate_flac", scope="module")
def fixture_audio_rate_flac():
    """fixture_audio_rate_flac"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_audio",
        "ZASFX_ADSR_no_sustain.flac",
    )

    args = path
    func = lambda args: tfio.IOTensor.graph(tf.int16).from_audio(args).rate
    expected = tf.constant(44100)

    return args, func, expected, np.array_equal


@pytest.fixture(name="audio_mp3", scope="module")
def fixture_audio_mp3():
    """fixture_audio_mp3"""
    # l1-fl6.bit was taken from minimp3
    # l1-fl6.raw is the converted, through minimp3
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_audio", "l1-fl6.bit"
    )
    raw_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_audio", "l1-fl6.raw"
    )
    raw = np.fromfile(raw_path, np.int16)
    raw = raw.reshape([-1, 2])
    value = tf.cast(raw, tf.float32) / 32768.0

    args = path
    func = lambda args: tfio.IOTensor.graph(tf.float32).from_audio(args)
    expected = value
    equal = lambda a, b: np.allclose(a, b, atol=0.00005)

    return args, func, expected, equal


@pytest.fixture(name="audio_rate_mp3", scope="module")
def fixture_audio_rate_mp3():
    """fixture_audio_rate_mp3"""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_audio", "l1-fl6.bit"
    )

    args = path
    func = lambda args: tfio.IOTensor.graph(tf.int16).from_audio(args).rate
    expected = tf.constant(44100)

    return args, func, expected, np.array_equal


@pytest.fixture(name="kafka")
def fixture_kafka():
    """fixture_kafka"""

    args = "test"

    def func(q):
        v = tfio.IOTensor.from_kafka(q)
        return v

    expected = [("D" + str(i)).encode() for i in range(10)]

    return args, func, expected, np.array_equal


@pytest.fixture(name="hdf5", scope="module")
def fixture_hdf5(request):
    """fixture_hdf5"""
    import h5py  # pylint: disable=import-outside-toplevel

    tmp_path = tempfile.mkdtemp()
    filename = os.path.join(tmp_path, "test.h5")

    data = list(range(5000))

    with h5py.File(filename, "w") as f:
        f.create_dataset("float64", data=np.asarray(data, np.float64), dtype="f8")
    args = filename

    def func(args):
        return tfio.IOTensor.from_hdf5(args)("/float64")

    expected = np.asarray(data, np.float64).tolist()

    def fin():
        shutil.rmtree(tmp_path)

    request.addfinalizer(fin)

    return args, func, expected, np.array_equal


@pytest.fixture(name="hdf5_graph", scope="module")
def fixture_hdf5_graph(request):
    """fixture_hdf5_graph"""
    import h5py  # pylint: disable=import-outside-toplevel

    tmp_path = tempfile.mkdtemp()
    filename = os.path.join(tmp_path, "test.h5")

    data = list(range(5000))

    with h5py.File(filename, "w") as f:
        f.create_dataset("float64", data=np.asarray(data, np.float64), dtype="f8")
    args = filename

    def func(args):
        return tfio.IOTensor.from_hdf5(args, spec={"/float64": tf.float64})("/float64")

    expected = np.asarray(data, np.float64).tolist()

    def fin():
        shutil.rmtree(tmp_path)

    request.addfinalizer(fin)

    return args, func, expected, np.array_equal


@pytest.fixture(name="hdf5_scalar", scope="module")
def fixture_hdf5_scalar(request):
    """fixture_hdf5_scalar"""
    import h5py  # pylint: disable=import-outside-toplevel

    tmp_path = tempfile.mkdtemp()
    filename = os.path.join(tmp_path, "test.h5")

    with h5py.File(filename, "w") as f:
        f.create_dataset("int8", data=np.int8(123))
        f.create_dataset("int16", data=np.int16(123))
        f.create_dataset("int32", data=np.int32(123))
        f.create_dataset("int64", data=np.int64(123))
        f.create_dataset("float32", data=np.float32(1.23))
        f.create_dataset("float64", data=np.float64(1.23))
        f.create_dataset("complex64", data=np.complex64(12 + 3j))
        f.create_dataset("complex128", data=np.complex128(12 + 3j))
        f.create_dataset("string", data=np.dtype("<S5").type("D123D"))
    args = filename

    def func(args):
        """func"""
        i8 = tfio.IOTensor.from_hdf5(args)("/int8")
        i16 = tfio.IOTensor.from_hdf5(args)("/int16")
        i32 = tfio.IOTensor.from_hdf5(args)("/int32")
        i64 = tfio.IOTensor.from_hdf5(args)("/int64")
        f32 = tfio.IOTensor.from_hdf5(args)("/float32")
        f64 = tfio.IOTensor.from_hdf5(args)("/float64")
        c64 = tfio.IOTensor.from_hdf5(args)("/complex64")
        c128 = tfio.IOTensor.from_hdf5(args)("/complex128")
        ss = tfio.IOTensor.from_hdf5(args)("/string")
        return [i8, i16, i32, i64, f32, f64, c64, c128, ss]

    expected = [
        np.int8(123),
        np.int16(123),
        np.int32(123),
        np.int64(123),
        np.float32(1.23),
        np.float64(1.23),
        np.complex64(12 + 3j),
        np.complex128(12 + 3j),
        np.dtype("<S5").type("D123D"),
    ]

    def fin():
        shutil.rmtree(tmp_path)

    request.addfinalizer(fin)

    return args, func, expected, np.array_equal


@pytest.fixture(name="hdf5_multiple_dimension", scope="module")
def fixture_hdf5_multiple_dimension(request):
    """fixture_hdf5_multiple_dimension"""
    import h5py  # pylint: disable=import-outside-toplevel

    tmp_path = tempfile.mkdtemp()
    filename = os.path.join(tmp_path, "test.h5")

    data = [np.arange(i, i + 20) for i in range(20)]

    with h5py.File(filename, "w") as f:
        f.create_dataset("float64", data=np.asarray(data, np.float64), dtype="f8")
    args = filename

    def func(args):
        return tfio.IOTensor.from_hdf5(args)("/float64")

    expected = np.asarray(data, np.float64)

    def fin():
        shutil.rmtree(tmp_path)

    request.addfinalizer(fin)

    return args, func, expected, np.array_equal


@pytest.fixture(name="arrow", scope="module")
def fixture_arrow():
    """fixture_arrow"""
    import pyarrow as pa  # pylint: disable=import-outside-toplevel

    arr1 = pa.array(list(range(100)), pa.int32())
    arr2 = pa.array(list(range(100)), pa.int64())
    arr3 = pa.array([x * 1.1 for x in range(100)], pa.float32())
    table = pa.Table.from_arrays([arr1, arr2, arr3], ["a", "b", "c"])

    args = table
    column = "b"
    func = lambda t: tfio.IOTensor.from_arrow(t)(column)
    expected = table[column].to_pylist()

    return args, func, expected, np.array_equal


@pytest.fixture(name="arrow_graph", scope="module")
def fixture_arrow_graph():
    """fixture_arrow"""
    import pyarrow as pa  # pylint: disable=import-outside-toplevel

    arr1 = pa.array(list(range(100)), pa.int32())
    arr2 = pa.array(list(range(100)), pa.int64())
    arr3 = pa.array([x * 1.1 for x in range(100)], pa.float32())
    table = pa.Table.from_arrays([arr1, arr2, arr3], ["a", "b", "c"])

    args = ""
    spec = {"a": tf.int32, "b": tf.int64, "c": tf.float32}
    column = "b"

    func = lambda _: tfio.IOTensor.from_arrow(table, spec=spec)(column)
    expected = table.column(column).to_pylist()

    return args, func, expected, np.array_equal


# scalar is a special IOTensor that is alias to Tensor
@pytest.mark.parametrize(
    ("io_tensor_fixture"),
    [
        pytest.param("hdf5_scalar"),
    ],
    ids=[
        "hdf5",
    ],
)
def test_io_tensor_scalar(fixture_lookup, io_tensor_fixture):
    """test_io_tensor_scalar"""
    args, func, expected, equal = fixture_lookup(io_tensor_fixture)

    values = func(args)

    # Test to_tensor
    entries = [value.to_tensor() for value in values]
    assert len(entries) == len(expected)
    assert np.all([equal(v, e) for v, e in zip(entries, expected)])


# slice (__getitem__) is the most basic operation for IOTensor
@pytest.mark.parametrize(
    ("io_tensor_fixture"),
    [
        pytest.param("audio_wav"),
        pytest.param("audio_wav_s24"),
        pytest.param("audio_flac"),
        pytest.param("audio_vorbis"),
        pytest.param("audio_mp3"),
        pytest.param("hdf5"),
        pytest.param("kafka"),
        pytest.param("arrow"),
    ],
    ids=[
        "audio[wav]",
        "audio[wav|s24]",
        "audio[flac]",
        "audio[vorbis]",
        "audio[mp3]",
        "hdf5",
        "kafka",
        "arrow",
    ],
)
def test_io_tensor_slice(fixture_lookup, io_tensor_fixture):
    """test_io_tensor_slice"""
    args, func, expected, equal = fixture_lookup(io_tensor_fixture)

    io_tensor = func(args)

    # Test to_tensor
    entries = io_tensor.to_tensor()
    assert len(entries) == len(expected)
    assert equal(entries, expected)

    # Test __getitem__, use 7 to partition
    indices = list(range(0, len(expected), 7))
    for start, stop in list(zip(indices, indices[1:] + [len(expected)])):
        assert equal(io_tensor[start:stop], expected[start:stop])


# full slicing/index across multiple dimensions
@pytest.mark.parametrize(
    ("io_tensor_fixture"),
    [
        pytest.param("hdf5_multiple_dimension"),
    ],
    ids=[
        "hdf5",
    ],
)
def test_io_tensor_slice_multiple_dimension(fixture_lookup, io_tensor_fixture):
    """test_io_tensor_slice_multiple_dimension"""
    args, func, expected, equal = fixture_lookup(io_tensor_fixture)

    io_tensor = func(args)

    # Test __getitem__, use 7 to partition dimension 0, 11 for dimension 1
    indices_0 = list(range(0, len(expected), 7))
    for start_0, stop_0 in list(zip(indices_0, indices_0[1:] + [len(expected)])):
        indices_1 = list(range(0, len(expected[0]), 11))
        for start_1, stop_1 in list(zip(indices_1, indices_1[1:] + [len(expected[0])])):
            assert equal(
                io_tensor[start_0:stop_0, start_1:stop_1],
                expected[start_0:stop_0, start_1:stop_1],
            )


# slice (__getitem__) could also be inside dataset for GraphIOTensor
@pytest.mark.parametrize(
    ("io_tensor_fixture", "num_parallel_calls"),
    [
        pytest.param("audio_wav", None),
        pytest.param("audio_wav", 2),
        pytest.param("audio_wav_s24", None),
        pytest.param("audio_wav_s24", 2),
        pytest.param("audio_flac", None),
        pytest.param("audio_flac", 2),
        pytest.param("audio_vorbis", None),
        pytest.param("audio_vorbis", 2),
        pytest.param("audio_mp3", None),
        pytest.param("audio_mp3", 2),
        pytest.param("hdf5_graph", None),
        pytest.param("hdf5_graph", 2),
        pytest.param("kafka", None),
        pytest.param("kafka", 2),
        pytest.param("arrow_graph", None),
        pytest.param("arrow_graph", 2),
    ],
    ids=[
        "audio[wav]",
        "audio[wav]|2",
        "audio[wav|s24]",
        "audio[wav|s24]|2",
        "audio[flac]",
        "audio[flac]|2",
        "audio[vorbis]",
        "audio[vorbis]|2",
        "audio[mp3]",
        "audio[mp3]|2",
        "hdf5",
        "hdf5|2",
        "kafka",
        "kafka|2",
        "arrow_graph",
        "arrow_graph|2",
    ],
)
def test_io_tensor_slice_in_dataset(
    fixture_lookup, io_tensor_fixture, num_parallel_calls
):
    """test_io_tensor_slice_in_dataset"""
    args, func, expected, equal = fixture_lookup(io_tensor_fixture)

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
        assert equal(entries, expected)
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
        assert equal(entries, expected[:100])
        item += 1
    assert item == 2


# meta is supported for IOTensor
@pytest.mark.parametrize(
    ("io_tensor_fixture"),
    [
        pytest.param("audio_rate_wav"),
        pytest.param("audio_rate_wav_s24"),
        pytest.param("audio_rate_flac"),
        pytest.param("audio_rate_vorbis"),
        pytest.param("audio_rate_mp3"),
    ],
    ids=[
        "audio[rate][wav]",
        "audio[rate][wav|s24]",
        "audio[rate][flac]",
        "audio[rate][vorbis]",
        "audio[rate][mp3]",
    ],
)
def test_io_tensor_meta(fixture_lookup, io_tensor_fixture):
    """test_io_tensor_slice"""
    args, func, expected, equal = fixture_lookup(io_tensor_fixture)

    # Test meta data attached to IOTensor
    meta = func(args)
    assert equal(meta, expected)


# meta inside dataset is also supported for GraphIOTensor
@pytest.mark.parametrize(
    ("io_tensor_fixture"),
    [
        pytest.param("audio_rate_wav"),
        pytest.param("audio_rate_wav_s24"),
        pytest.param("audio_rate_flac"),
        pytest.param("audio_rate_vorbis"),
        pytest.param("audio_rate_mp3"),
    ],
    ids=[
        "audio[rate][wav]",
        "audio[rate][wav|s24]",
        "audio[rate][flac]",
        "audio[rate][vorbis]",
        "audio[rate][mp3]",
    ],
)
def test_io_tensor_meta_in_dataset(fixture_lookup, io_tensor_fixture):
    """test_io_tensor_slice"""
    args, func, expected, equal = fixture_lookup(io_tensor_fixture)

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
        assert equal(meta, expected)
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
        pytest.param("audio_wav_s24"),
        pytest.param("audio_flac"),
        pytest.param("audio_vorbis"),
        pytest.param("audio_mp3"),
        pytest.param("hdf5"),
        pytest.param("arrow"),
    ],
    ids=[
        "audio[wav]",
        "audio[wav|s24]",
        "audio[flac]",
        "audio[vorbis]",
        "audio[mp3]",
        "hdf5",
        "arrow",
    ],
)
def test_io_tensor_benchmark(benchmark, fixture_lookup, io_tensor_fixture):
    """test_io_tensor_benchmark"""
    args, func, expected, equal = fixture_lookup(io_tensor_fixture)

    def f(v):
        io_tensor = func(v)
        return io_tensor.to_tensor()

    entries = benchmark(f, args)

    assert equal(entries, expected)
