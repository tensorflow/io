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
"""Tests for VideoDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.platform import test

from tensorflow_io.video.python.ops import video_dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import resource_loader


class VideoDatasetTest(test.TestCase):

  def test_video_dataset(self):
    """Test case for VideoDataset."""
    filename = os.path.join(resource_loader.get_data_files_path(),
                            "testdata", "small.mp4")

    filenames = constant_op.constant([filename], dtypes.string)
    num_repeats = 2

    dataset = video_dataset_ops.VideoDataset(filenames).repeat(
        num_repeats)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):  # Dataset is repeated.
        for _ in range(166): # 166 frames
          v = sess.run(get_next)
          self.assertAllEqual(v.shape, (320, 560, 3))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
