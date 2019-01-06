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
"""Tests for Image Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.platform import test

from tensorflow_io.image.python.ops import image_dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import resource_loader


class ImageDatasetTest(test.TestCase):

  def test_webp_file_dataset(self):
    """Test case for WebPDataset.
    """
    width = 400
    height = 301
    channel = 4
    png_file = os.path.join(resource_loader.get_data_files_path(),
                            "testdata", "sample.png")
    with open(png_file, 'rb') as f:
      png_contents = f.read()
    with self.cached_session():
      image_op = image_ops.decode_png(png_contents, channels=channel)
      image = image_op.eval()
      self.assertEqual(image.shape, (height, width, channel))

    filename = os.path.join(resource_loader.get_data_files_path(),
                            "testdata", "sample.webp")

    filenames = constant_op.constant([filename], dtypes.string)
    num_repeats = 2

    dataset = image_dataset_ops.WebPDataset(filenames).repeat(
        num_repeats)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):  # Dataset is repeated.
        v = sess.run(get_next)
        self.assertAllEqual(image, v)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
