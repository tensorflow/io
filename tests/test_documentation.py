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
"""Tests for MNIST Dataset with tf.keras."""


import os
import sys
import pytest

if sys.platform == "darwin":
    pytest.skip(
        "Documentation test only need to be run on Linux, skip macOS",
        allow_module_level=True,
    )


def extract_block(filename, lang):
    """extract_block"""
    source = ""
    with open(filename) as f:
        hit = -1
        for line in f:
            if hit < 0:
                if line.strip().startswith("```" + lang):
                    hit = line.find("```" + lang)
            else:
                if line.strip().startswith("```") and line.find("```") == hit:
                    break
                source += line
    return source


def test_readme():
    """test_readme"""
    # Note: From README.md
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "README.md"
    )
    source = extract_block(filename, "python")
    exec(source, globals())  # pylint: disable=exec-used
