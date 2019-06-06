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
"""Tests for Text Input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pytest
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow import errors # pylint: disable=wrong-import-position
import tensorflow_io.text as text_io # pylint: disable=wrong-import-position

def test_text_input():
  """test_text_input
  """
  text_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_text", "lorem.txt")
  with open(text_filename, 'rb') as f:
    lines = [line.strip() for line in f]
  text_filename = "file://" + text_filename

  gzip_text_filename = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "test_text", "lorem.txt.gz")
  gzip_text_filename = "file://" + gzip_text_filename

  lines = lines * 3
  filenames = [text_filename, gzip_text_filename, text_filename]
  dataset = text_io.TextDataset(filenames, batch=2)
  iterator = dataset.make_initializable_iterator()
  init_op = iterator.initializer
  get_next = iterator.get_next()
  with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    for i in range(0, len(lines) - 2, 2):
      v = sess.run(get_next)
      assert lines[i] == v[0]
      assert lines[i + 1] == v[1]
    v = sess.run(get_next)
    assert lines[len(lines) - 1] == v[0]
    with pytest.raises(errors.OutOfRangeError):
      sess.run(get_next)

if __name__ == "__main__":
  test.main()
