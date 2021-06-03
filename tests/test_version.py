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
"""Tests for version string"""


import tensorflow_io as tfio
from tensorflow_io.python.ops import version_ops


def test_version():
    """test_version"""
    assert tfio.__version__ == version_ops.version
    assert tfio.version.VERSION == version_ops.version


if __name__ == "__main__":
    test.main()
