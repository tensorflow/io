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
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio


@pytest.fixture(name="video_path", scope="module")
def fixture_video_path():
    return "file://" + os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_video", "small.mp4"
    )


@pytest.fixture(name="audio_path", scope="module")
def fixture_audio_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_audio", "mono_10khz.wav"
    )


def test_ffmpeg_io_tensor_audio(audio_path):
    """test_ffmpeg_io_tensor_audio"""
    audio = tfio.IOTensor.from_audio(audio_path)
    ffmpeg = tfio.IOTensor.from_ffmpeg(audio_path)
    ffmpeg = ffmpeg("a:0")
    assert audio.dtype == ffmpeg.dtype
    assert audio.rate == ffmpeg.rate
    assert audio.shape[1] == ffmpeg.shape[1]
    assert np.all(audio[0:5760].numpy() == ffmpeg[0:5760].numpy())
    assert len(audio) == len(ffmpeg)


# Disable as the mkv file is large. Run locally
# by pulling test file while is located in:
# https://github.com/Matroska-Org/matroska-test-files/blob/master/test_files/test5.mkv
def _test_ffmpeg_io_tensor_mkv(video_path):
    """test_ffmpeg_io_tensor_mkv"""
    mkv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_video", "test5.mkv"
    )
    mkv = tfio.IOTensor.from_ffmpeg(mkv_path)
    assert mkv("a:0").shape.as_list() == [None, 2]
    assert mkv("a:0").dtype == tf.float32
    assert mkv("a:0").rate == 48000
    assert mkv("s:0").shape.as_list() == [None]
    assert mkv("s:0").dtype == tf.string
    assert mkv("s:0")[0] == ["...the colossus of Rhodes!\r\n"]
    assert mkv("s:0")[1] == ["No!\r\n"]
    assert mkv("s:0")[2] == [
        "The colossus of Rhodes\\Nand it is here just for you Proog.\r\n"
    ]
    assert mkv("s:0")[3] == ["It is there...\r\n"]
    assert mkv("s:0")[4] == ["I'm telling you,\\NEmo...\r\n"]

    video = tfio.IOTensor.from_ffmpeg(video_path)
    assert video("v:0").shape.as_list() == [None, 320, 560, 3]
    assert video("v:0").dtype == tf.uint8
    assert len(video("v:0")) == 166
    assert video("v:0").to_tensor().shape == [166, 320, 560, 3]


def test_ffmpeg_decode_video(video_path):
    """test_ffmpeg_decode_video"""
    content = tf.io.read_file(video_path)
    video = tfio.experimental.ffmpeg.decode_video(content, 0)
    assert video.shape == [166, 320, 560, 3]
    assert video.dtype == tf.uint8
    assert np.abs(video[0] - video[-1]).sum() > 0


def test_ffmpeg_decode_video_invalid_content():
    """test_ffmpeg_decode_video_invalid_content"""
    content = tf.constant(b"bad-video")
    with pytest.raises(tf.errors.InvalidArgumentError):
        tfio.experimental.ffmpeg.decode_video(content, 0)


def test_ffmpeg_decode_video_invalid_index(video_path):
    """test_ffmpeg_decode_video_invalid_index"""
    content = tf.io.read_file(video_path)
    with pytest.raises(tf.errors.InvalidArgumentError):
        tfio.experimental.ffmpeg.decode_video(content, 1)


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS fails now")
def test_video_predict(video_path):
    """test_video_predict"""
    model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")
    x = (
        tfio.IODataset.from_ffmpeg(video_path, "v:0")
        .batch(1)
        .map(
            lambda x: tf.keras.applications.resnet50.preprocess_input(
                tf.image.resize(x, (224, 224))
            )
        )
    )
    y = model.predict(x)
    p = tf.keras.applications.resnet50.decode_predictions(y, top=1)
    assert len(p) == 166
