# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for file system configuration API"""

import pytest

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_io_gcs_filesystem


def test_filesystem_configuration():
    """Test case for configuration"""
    with pytest.raises(tf.errors.UnimplementedError) as e:
        tfio.experimental.filesystem.set_configuration("gs", "123", "456")
    assert (
        "SetConfiguration not implemented for gcs ('gs://') file system: name = 123, value = 456"
        in str(e.value)
    )
