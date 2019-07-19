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
"""Tests for DICOM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_io.dicom as dicom_io  # pylint: disable=wrong-import-position

import os
import pytest

tf.compat.v1.disable_eager_execution()


def test_dicom_input():
    """test_dicom_input
    """
    _ = dicom_io.decode_dicom_data
    _ = dicom_io.decode_dicom_image
    _ = dicom_io.tags


def test_decode_dicom_image():
    """test_decode_dicom_image
    """
    files = [
        ('OT-MONO2-8-colon.dcm', (1, 512, 512, 1)),
        ('CR-MONO1-10-chest.dcm', (1, 440, 440, 1)),
        ('CT-MONO2-16-ort.dcm', (1, 512, 512, 1)),
        ('MR-MONO2-16-head.dcm', (1, 256, 256, 1)),
        ('US-RGB-8-epicard.dcm', (1, 480, 640, 3)),
        ('CT-MONO2-8-abdo.dcm', (1, 512, 512, 1)),
        ('MR-MONO2-16-knee.dcm', (1, 256, 256, 1)),
        ('OT-MONO2-8-hip.dcm', (1, 512, 512, 1)),
        ('US-RGB-8-esopecho.dcm', (1, 120, 256, 3)),
        ('CT-MONO2-16-ankle.dcm', (1, 512, 512, 1)),
        ('MR-MONO2-12-an2.dcm', (1, 256, 256, 1)),
        ('MR-MONO2-8-16x-heart.dcm', (16, 256, 256, 1)),
        ('OT-PAL-8-face.dcm', (1, 480, 640, 3)),
        ('XA-MONO2-8-12x-catheter.dcm', (12, 512, 512, 1)),
        ('CT-MONO2-16-brain.dcm', (1, 512, 512, 1)),
        ('NM-MONO2-16-13x-heart.dcm', (13, 64, 64, 1)),
        ('US-MONO2-8-8x-execho.dcm', (8, 120, 128, 1)),
        ('CT-MONO2-16-chest.dcm', (1, 400, 512, 1)),
        ('MR-MONO2-12-shoulder.dcm', (1, 1024, 1024, 1)),
        ('OT-MONO2-8-a7.dcm', (1, 512, 512, 1)),
        ('US-PAL-8-10x-echo.dcm', (10, 430, 600, 3)),
    ]
    for fname, im_shape in files:
        dcm_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_dicom",
            fname
        )

        G1 = tf.Graph()

        with G1.as_default():
            file_contents = tf.io.read_file(filename=dcm_file)
            dcm_image = dicom_io.decode_dicom_image(
                contents=file_contents,
                dtype=tf.float32,
                on_error='strict',
                scale='auto',
                color_dim=True,
            )

        sess = tf.Session(graph=G1)
        dcm_image_np = sess.run(dcm_image)

        assert dcm_image_np.shape == im_shape


if __name__ == "__main__":
    test.main()
