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
"""Tests for GRPC Input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow
tensorflow.compat.v1.disable_eager_execution()

from tensorflow import errors # pylint: disable=wrong-import-position
import tensorflow_io.grpc as grpc_io # pylint: disable=wrong-import-position

def test_grpc_input():
  """test_grpc_input"""
  data = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  dataset = grpc_io.GRPCDataset.from_numpy(data, batch=2)
  iterator = dataset.make_initializable_iterator()
  init_op = iterator.initializer
  get_next = iterator.get_next()
  with tensorflow.compat.v1.Session() as sess:
    sess.run(init_op)
    v = sess.run(get_next)
    assert np.alltrue(data[0:2] == v)
    v = sess.run(get_next)
    assert np.alltrue(data[2:3] == v)
    with pytest.raises(errors.OutOfRangeError):
      sess.run(get_next)

if __name__ == "__main__":
  test.main()
