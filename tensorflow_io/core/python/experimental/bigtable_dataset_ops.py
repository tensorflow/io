# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""BigTableIOLayer"""


import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops
from tensorflow_io.bigtable.python.ops import bigtable_api;

class BigTableIODataset(tf.data.Dataset):
    """BigTableIODataset"""

    def __init__(self, project=None, instance=None, table=None):
        """BigTableIODataset."""
        return;
