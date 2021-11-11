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
"""Tests for ImageIOTensor."""

import os
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio


def test_tiff_io_tensor():
    """Test case for TIFFImageIOTensor"""
    width = 560
    height = 320
    channels = 4

    images = []
    for filename in [
        "small-00.png",
        "small-01.png",
        "small-02.png",
        "small-03.png",
        "small-04.png",
    ]:
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test_image", filename
            ),
            "rb",
        ) as f:
            png_contents = f.read()
        image_v = tf.image.decode_png(png_contents, channels=channels)
        assert image_v.shape == [height, width, channels]
        images.append(image_v)

    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "small.tiff"
    )

    tiff = tfio.IOTensor.from_tiff(filename)
    assert tiff.keys == list(range(5))
    for i in tiff.keys:
        assert np.all(images[i].numpy() == tiff(i).to_tensor().numpy())


def test_decode_webp():
    """Test case for decode_webp."""
    width = 400
    height = 301
    channel = 4
    png_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "sample.png"
    )
    with open(png_file, "rb") as f:
        png_contents = f.read()
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "sample.webp"
    )
    with open(filename, "rb") as f:
        webp_contents = f.read()

    png = tf.image.decode_png(png_contents, channels=channel)
    assert png.shape == (height, width, channel)

    webp_v = tfio.image.decode_webp(webp_contents)
    assert webp_v.shape == (height, width, channel)

    assert np.all(webp_v == png)


def test_tiff_file_dataset():
    """Test case for TIFFDataset."""
    width = 560
    height = 320
    channels = 4

    images = []
    for filename in [
        "small-00.png",
        "small-01.png",
        "small-02.png",
        "small-03.png",
        "small-04.png",
    ]:
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test_image", filename
            ),
            "rb",
        ) as f:
            png_contents = f.read()
        image_v = tf.image.decode_png(png_contents, channels=channels)
        assert image_v.shape == [height, width, channels]
        images.append(image_v)

    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "small.tiff"
    )

    num_repeats = 2

    dataset = tfio.experimental.IODataset.from_tiff(filename).repeat(num_repeats)
    i = 0
    for v in dataset:
        np.all(images[i % 5] == v)
        i += 1
    assert i == 10


def test_draw_bounding_box():
    """Test case for draw_bounding_box."""
    width = 560
    height = 320
    channels = 4

    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_image", "small-00.png"
        ),
        "rb",
    ) as f:
        png_contents = f.read()
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_image", "small-bb.png"
        ),
        "rb",
    ) as f:
        ex_png_contents = f.read()
    ex_image_p = tf.image.decode_png(ex_png_contents, channels=channels)
    # TODO: Travis seems to have issues with different rendering. Skip for now.
    # ex_image_v = ex_image_p.eval()
    _ = tf.expand_dims(ex_image_p, 0)

    bb = [[[0.1, 0.2, 0.5, 0.9]]]
    image_v = tf.image.decode_png(png_contents, channels=channels)
    assert image_v.shape == (height, width, channels)
    image_p = tf.image.convert_image_dtype(image_v, tf.float32)
    image_p = tf.expand_dims(image_p, 0)
    bb_image_p = tfio.experimental.image.draw_bounding_boxes(
        image_p, bb, ["hello world!"]
    )
    # TODO: Travis seems to have issues with different rendering. Skip for now.
    # self.assertAllEqual(bb_image_v, ex_image_v)
    _ = tf.image.convert_image_dtype(bb_image_p, tf.uint8)


def test_decode_ppm():
    """Test case for decode_ppm"""
    ppm_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "r-1316653631.481244-81973200.ppm",
    )
    png_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "r-1316653631.481244-81973200.png",
    )
    ppm = tfio.experimental.image.decode_pnm(tf.io.read_file(ppm_file))
    png = tf.image.decode_png(tf.io.read_file(png_file))
    assert np.all(ppm.numpy() == png.numpy())

    pgm_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "d-1316653631.269651-68451027.pgm",
    )
    png_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "d-1316653631.269651-68451027.png",
    )
    pgm = tfio.experimental.image.decode_pnm(tf.io.read_file(pgm_file), dtype=tf.uint16)
    png = tf.image.decode_png(tf.io.read_file(png_file), dtype=tf.uint16)
    assert np.all(pgm.numpy() == png.numpy())


def test_encode_bmp():
    """Test case for encode_bmp."""
    width = 51
    height = 26
    channels = 3
    bmp_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "lena.bmp"
    )
    with open(bmp_file, "rb") as f:
        bmp_contents = f.read()
    image_v = tf.image.decode_bmp(bmp_contents)
    assert image_v.shape == [height, width, channels]
    bmp_encoded = tfio.image.encode_bmp(image_v)
    image_e = tf.image.decode_bmp(bmp_encoded)
    assert np.all(image_v.numpy() == image_e.numpy())


def test_decode_exif():
    """Test case for decode_exif."""
    jpeg_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "down-mirrored.jpg"
    )
    exif = tfio.experimental.image.decode_jpeg_exif(tf.io.read_file(jpeg_file))
    assert exif == 4


def test_openexr_io_tensor():
    """Test case for OpenEXRIOTensor"""
    # image from http://gl.ict.usc.edu/Data/HighResProbes/
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "glacier.exr"
    )

    exr_shape, exr_dtype, exr_channel = tfio.experimental.image.decode_exr_info(
        tf.io.read_file(filename)
    )
    assert np.all(exr_shape == [1024, 2048])
    assert np.all(exr_dtype.numpy() == [tf.float16, tf.float16, tf.float16])
    assert np.all(exr_channel == ["B", "G", "R"])

    exr_0_b = tfio.experimental.image.decode_exr(
        tf.io.read_file(filename), 0, "B", tf.float16
    )
    exr_0_g = tfio.experimental.image.decode_exr(
        tf.io.read_file(filename), 0, "G", tf.float16
    )
    exr_0_r = tfio.experimental.image.decode_exr(
        tf.io.read_file(filename), 0, "R", tf.float16
    )

    exr = tfio.experimental.IOTensor.from_exr(filename)
    assert exr.keys == [0]
    assert exr(0).columns == ["B", "G", "R"]

    assert exr(0)("B").dtype == tf.float16
    assert exr(0)("G").dtype == tf.float16
    assert exr(0)("R").dtype == tf.float16

    assert exr(0)("B").shape == [1024, 2048]
    assert exr(0)("G").shape == [1024, 2048]
    assert exr(0)("R").shape == [1024, 2048]

    b = exr(0)("B").to_tensor()
    g = exr(0)("G").to_tensor()
    r = exr(0)("R").to_tensor()

    assert b.shape == [1024, 2048]
    assert g.shape == [1024, 2048]
    assert r.shape == [1024, 2048]

    assert b.dtype == tf.float16
    assert g.dtype == tf.float16
    assert r.dtype == tf.float16

    rgb = tf.stack([r, g, b], axis=2)
    rgb = tf.image.convert_image_dtype(rgb, tf.uint8)
    _ = tf.image.encode_png(rgb)
    # TODO: compare with generated png
    # tf.io.write_file('sample.png', png)

    assert np.all(b == exr_0_b)
    assert np.all(g == exr_0_g)
    assert np.all(r == exr_0_r)


def test_decode_hdr():
    """Test case for decode_hdr"""
    # image from http://gl.ict.usc.edu/Data/HighResProbes/
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "glacier.hdr"
    )

    contents = tf.io.read_file(filename)
    hdr = tfio.experimental.image.decode_hdr(contents)
    assert hdr.dtype == tf.float32
    assert hdr.shape == [1024, 2048, 3]
    rgb = tf.image.convert_image_dtype(hdr, tf.uint8)
    _ = tf.image.encode_png(rgb)
    # TODO: compare with generated png
    # tf.io.write_file('sample.png', png)


def test_decode_tiff_geotiff():
    """Test case for decode_tiff_geotiff"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "GeogToWGS84GeoKey5.tif",
    )
    shape, dtype = tfio.experimental.image.decode_tiff_info(tf.io.read_file(filename))
    # only one item for now
    assert np.all(shape.shape == [1, 3])
    assert np.all(dtype.shape == [1])
    assert np.all(shape[0] == [101, 101, 1])
    assert np.all(dtype[0].numpy() == tf.uint8)

    image = tfio.experimental.image.decode_tiff(tf.io.read_file(filename))

    png_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "GeogToWGS84GeoKey5.png",
    )
    png_image = tf.image.decode_png(tf.io.read_file(png_filename))

    image = image[:, :, 0:3]
    assert np.all(png_image.numpy() == image.numpy())


def test_decode_nv12():
    """Test case for decode_nv12"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "Jelly-Beans.nv12"
    )
    png_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "Jelly-Beans.nv12.png"
    )
    png = tf.image.decode_png(tf.io.read_file(png_filename))

    contents = tf.io.read_file(filename)
    rgb = tfio.experimental.image.decode_nv12(contents, size=[256, 256])
    assert rgb.dtype == tf.uint8
    assert rgb.shape == [256, 256, 3]
    assert np.all(rgb == png)


def test_decode_yuy2():
    """Test case for decode_yuy2"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "Jelly-Beans.yuy2"
    )
    png_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "Jelly-Beans.yuy2.png"
    )
    png = tf.image.decode_png(tf.io.read_file(png_filename))

    contents = tf.io.read_file(filename)
    rgb = tfio.experimental.image.decode_yuy2(contents, size=[256, 256])
    assert rgb.dtype == tf.uint8
    assert rgb.shape == [256, 256, 3]
    assert np.all(rgb == png)


def test_decode_avif():
    """Test case for decode_avif"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "kodim03_yuv420_8bpc.avif",
    )
    png_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "kodim03_yuv420_8bpc.png",
    )
    png = tf.image.decode_png(tf.io.read_file(png_filename))

    contents = tf.io.read_file(filename)
    rgb = tfio.experimental.image.decode_avif(contents)
    assert rgb.dtype == tf.uint8
    assert rgb.shape == png.shape
    assert np.all(rgb == png)


def test_decode_tiff_multipage():
    """Test case for decode_tiff_multipage"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "multipage_tiff_example.tif",
    )
    shape, dtype = tfio.experimental.image.decode_tiff_info(tf.io.read_file(filename))
    assert np.all(shape.shape == [10, 3])
    assert np.all(dtype.shape == [10])
    for i in range(10):
        assert np.array_equal(shape[i], [600, 800, 3])
        assert dtype[i].numpy() == tf.uint8
        # TODO: validate content
        image = tfio.experimental.image.decode_tiff(tf.io.read_file(filename), index=i)


def test_decode_jp2():
    """Test case for decode_jp2"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "Jelly-Beans.jp2",
    )
    png_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "Jelly-Beans.jp2.png",
    )
    png = tf.image.decode_png(tf.io.read_file(png_filename))

    contents = tf.io.read_file(filename)
    rgb = tfio.experimental.image.decode_jp2(contents)
    assert rgb.dtype == tf.uint8
    assert rgb.shape == png.shape
    assert np.all(rgb == png)


def test_decode_jp2_uint16():
    """Test case for decode_jp2_uint16"""
    # The image is generated from:
    # data = np.asarray(range(512), np.uint16)
    # data = np.broadcast_to(data, [512, 512]) * 128
    # jp2 = glymur.Jp2k('img.jp2', data=data, cratios=[20, 10, 1])
    data = np.asarray(range(512), np.uint16)
    data = np.broadcast_to(data, [512, 512]) * 128
    data = np.reshape(data, [512, 512, 1])

    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "img.jp2",
    )

    contents = tf.io.read_file(filename)
    rgb = tfio.experimental.image.decode_jp2(contents, dtype=tf.uint16)
    assert rgb.dtype == tf.uint16
    assert rgb.shape == data.shape
    assert np.array_equal(rgb, data)


def test_encode_gif():
    """Test case for encode_gif."""

    # Image is taken from WIKI (Newton's Cradle: Newtons_cradle_animation_book_2.gif):
    # https://en.wikipedia.org/wiki/GIF
    batch = 36
    height = 360
    width = 480
    channel = 3

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "cradle.gif"
    )

    image = tf.image.decode_gif(tf.io.read_file(path))
    assert image.shape == (batch, height, width, channel)
    gif = tfio.image.encode_gif(image)
    encoded = tf.image.decode_gif(gif)
    assert np.allclose(image, encoded, atol=8.0)


def test_decode_tiff_16bit():
    """Test case for 16 bit tiff"""
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_image",
        "IXMtest_A01_s1_w164FBEEF7-F77C-4892-86F5-72D0160D4FB2.tif",
    )
    shape, dtype = tfio.experimental.image.decode_tiff_info(tf.io.read_file(filename))
    # only one item for now
    assert np.all(shape.shape == [1, 3])
    assert np.all(dtype.shape == [1])
    assert np.all(shape[0] == [520, 696, 1])
    assert np.all(dtype[0].numpy() == tf.uint16)

    image = tfio.experimental.image.decode_tiff(tf.io.read_file(filename))
    assert image.shape == [520, 696, 4]


if __name__ == "__main__":
    test.main()
