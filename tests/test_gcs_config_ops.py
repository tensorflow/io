# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the gcs_config_ops."""


import sys
import pytest

import tensorflow as tf

from tensorflow.python.platform import test
from tensorflow_io import gcs

tf_v1 = tf.version.VERSION.startswith('1')

class GcsConfigOpsTest(test.TestCase):
  """GCS Config OPS test"""

  @pytest.mark.skipif(sys.platform == "darwin", reason=None)
  def test_set_block_cache(self):
    """test_set_block_cache"""
    cfg = gcs.BlockCacheParams(max_bytes=1024*1024*1024)
    if tf_v1:
      with tf.Session() as session:
        gcs.configure_gcs(session,
                          credentials=None,
                          block_cache=cfg,
                          device=None)
    else:
      gcs.configure_gcs(block_cache=cfg)


if __name__ == '__main__':
  test.main()
