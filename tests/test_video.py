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
"""test ffmpeg dataset"""

import os
import sys
import pytest

import tensorflow as tf
import tensorflow_io as tfio


@pytest.fixture(name="video_data", scope="module")
def fixture_video_data():
    """fixture_video_data"""
    path = "file://" + os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_video", "small.mp4"
    )
    # TODO: get raw value
    value = tf.zeros((166, 320, 560, 3), tf.uint8)
    return path, value


@pytest.mark.parametrize(
    ("io_dataset_func"),
    [
        pytest.param(
            lambda f: tfio.IODataset.graph(tf.uint8).from_ffmpeg(f, "v:0"),
            marks=[
                pytest.mark.skipif(
                    sys.platform == "darwin", reason="TODO: !!!pytest-xdist!!! on macOS"
                ),
            ],
        ),
        pytest.param(
            lambda f: tfio.IODataset.from_ffmpeg(f, "v:0"),
            marks=[
                pytest.mark.skipif(
                    True,  # sys.platform == "darwin",
                    reason="macOS does not support FFmpeg",
                ),
            ],
        ),
    ],
    ids=["from_ffmpeg", "from_ffmpeg(eager)"],
)
def test_video_io_dataset(video_data, io_dataset_func):
    """test_video_io_dataset"""
    video_path, video_value = video_data

    video_dataset = io_dataset_func(video_path)

    i = 0
    for value in video_dataset:
        assert video_value[i].shape == value.shape
        i += 1
    assert i == 166

    video_dataset = io_dataset_func(video_path).batch(2)

    i = 0
    for value in video_dataset:
        assert video_value[i : i + 2].shape == value.shape
        i += 2
    assert i == 166


@pytest.mark.parametrize(
    ("io_dataset_func"),
    [
        pytest.param(
            lambda f: tfio.IODataset.graph(tf.uint8).from_ffmpeg(f, "v:0"),
            marks=[
                pytest.mark.skipif(
                    sys.platform == "darwin", reason="TODO: !!!pytest-xdist!!! on macOS"
                ),
            ],
        ),
    ],
    ids=["from_ffmpeg"],
)
def test_video_io_dataset_with_dataset(video_data, io_dataset_func):
    """test_video_io_dataset_with_dataset"""
    video_path, video_value = video_data

    filename_dataset = tf.data.Dataset.from_tensor_slices([video_path, video_path])
    position_dataset = tf.data.Dataset.from_tensor_slices(
        [tf.constant(50, tf.int64), tf.constant(100, tf.int64)]
    )

    dataset = tf.data.Dataset.zip((filename_dataset, position_dataset))

    # Note: @tf.function is actually not needed, as tf.data.Dataset
    # will automatically wrap the `func` into a graph anyway.
    # The following is purely for explanation purposes.
    # Return: an embedded dataset (in an outer dataset) for position:position+100
    @tf.function
    def func(filename, position):
        video_dataset = io_dataset_func(filename)
        return video_dataset.skip(position).take(10)

    dataset = dataset.map(func)

    item = 0
    # Notice video_dataset in dataset:
    for video_dataset in dataset:
        position = 50 if item == 0 else 100
        i = 0
        for value in video_dataset:
            assert video_value[position + i].shape == value.shape
            i += 1
        assert i == 10
        item += 1
    assert item == 2
