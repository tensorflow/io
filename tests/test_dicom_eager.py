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

import os
import pytest

import tensorflow as tf
import tensorflow_io as tfio

# The DICOM sample files must be downloaded befor running the tests
#
# To download the DICOM samples:
# $ bash dicom_samples.sh download
# $ bash dicom_samples.sh extract
#
# To remopve the DICOM samples:
# $ bash dicom_samples.sh clean_dcm
#
# To remopve all the downloaded files:
# $ bash dicom_samples.sh clean_all


def test_dicom_input():
  """test_dicom_input
  """
  _ = tfio.image.decode_dicom_data
  _ = tfio.image.decode_dicom_image
  _ = tfio.image.dicom_tags


@pytest.mark.parametrize(
    'fname, exp_shape',
    [
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
)
def test_decode_dicom_image(fname, exp_shape):
  """test_decode_dicom_image
  """

  dcm_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_dicom",
      fname
  )

  file_contents = tf.io.read_file(filename=dcm_path)

  dcm_image = tfio.image.decode_dicom_image(
      contents=file_contents,
      dtype=tf.float32,
      on_error='strict',
      scale='auto',
      color_dim=True,
  )
  assert dcm_image.numpy().shape == exp_shape


@pytest.mark.parametrize(
    'fname, tag, exp_value',
    [
        (
            'OT-MONO2-8-colon.dcm',
            tfio.image.dicom_tags.StudyInstanceUID,
            b'1.3.46.670589.17.1.7.1.1.16'
        ),
        (
            'OT-MONO2-8-colon.dcm',
            tfio.image.dicom_tags.Rows,
            b'512'
        ),
        (
            'OT-MONO2-8-colon.dcm',
            tfio.image.dicom_tags.Columns,
            b'512'
        ),
        (
            'OT-MONO2-8-colon.dcm',
            tfio.image.dicom_tags.SamplesperPixel,
            b'1'
        ),
        (
            'US-PAL-8-10x-echo.dcm',
            tfio.image.dicom_tags.StudyInstanceUID,
            b'999.999.3859744'
        ),
        (
            'US-PAL-8-10x-echo.dcm',
            tfio.image.dicom_tags.SeriesInstanceUID,
            b'999.999.94827453'
        ),
        (
            'US-PAL-8-10x-echo.dcm',
            tfio.image.dicom_tags.NumberofFrames,
            b'10'
        ),
        (
            'US-PAL-8-10x-echo.dcm',
            tfio.image.dicom_tags.Rows,
            b'430'
        ),
        (
            'US-PAL-8-10x-echo.dcm',
            tfio.image.dicom_tags.Columns,
            b'600'
        ),
    ]
)
def test_decode_dicom_data(fname, tag, exp_value):
  """test_decode_dicom_data
  """

  dcm_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_dicom",
      fname
  )

  file_contents = tf.io.read_file(filename=dcm_path)

  dcm_data = tfio.image.decode_dicom_data(
      contents=file_contents,
      tags=tag
  )

  assert dcm_data.numpy() == exp_value

def test_dicom_image_shape():
  """test_decode_dicom_image"""

  dcm_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "test_dicom",
      'US-PAL-8-10x-echo.dcm'
  )

  dataset = tf.data.Dataset.from_tensor_slices([dcm_path])
  dataset = dataset.map(tf.io.read_file)
  dataset = dataset.map(
      lambda e: tfio.image.decode_dicom_image(e, dtype=tf.uint16))
  dataset = dataset.map(
      lambda e: tf.image.resize(e, (224, 224)))


if __name__ == "__main__":
  test.main()
